import torch

from torch.autograd import Variable

from onmt.translate.decode_strategy import DecodeStrategy
from onmt.translate import penalties

from onmt.inputters.dataset_base import _dynamic_dict

def get_LM_prob(lm_model, s):
    return lm_model.score(s, bos = True, eos = True)

def get_LM_perplexity(lm_model, s):
    return lm_model.perplexity(s)

class BeamSearch(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        mb_device (torch.device or str): See base ``device``.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best, mb_device,
                 global_scorer, min_length, max_length, return_attention,
                 block_ngram_repeat, exclusion_tokens, memory_lengths,
                 stepwise_penalty, ratio, 
                 src_vocab=None,
                 tgt_vocab=None,
                 src_vocabs=None,
                 lm_model=None,
                 LM_rerank_score_demultiplier=1.0,
                 alternate_scorer=False,
                 all_source=None):
        super(BeamSearch, self).__init__(
            pad, bos, eos, batch_size, mb_device, beam_size, min_length,
            block_ngram_repeat, exclusion_tokens, return_attention,
            max_length)
        # beam parameters
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size
        self.ratio = ratio

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.best_scores = torch.full([batch_size], -1e10, dtype=torch.float,
                                      device=mb_device)

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long,
            device=mb_device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=mb_device
        ).repeat(batch_size)
        self.select_indices = None
        self._memory_lengths = memory_lengths

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size),
                                       dtype=torch.float, device=mb_device)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long,
                                    device=mb_device)
        self._batch_index = torch.empty([batch_size, beam_size],
                                        dtype=torch.long, device=mb_device)
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = (
                stepwise_penalty and self.global_scorer.has_cov_pen)
        self._vanilla_cov_pen = (
            not stepwise_penalty and self.global_scorer.has_cov_pen)
        self._cov_pen = self.global_scorer.has_cov_pen



        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_vocabs = src_vocabs
        
        ## NOTE: multiple runs show that unk is at location 0 and the model still outupts unk sometimes
        ## Will hardcode the unk probablitites to close to 0
        # print("Unk id:", self._src_vocabs.stoi['<unk>'])
        # for i, word in enumerate(self._src_vocabs.itos):
        #     if word == "<unk>":
        #         print(i)
        #         print(word)
        # exit()
        # print("SRC vocabs:")
        # print(self._src_vocabs)
        # print(len(self._src_vocabs))
        self.lm_model = lm_model
        self.LM_rerank_score_demultiplier = LM_rerank_score_demultiplier
        self._bos = bos
        self._eos = eos
        self.source_text = all_source
        self.answer_text = ' '.join(self.source_text[self.source_text.index("|||") + 1:])
        # print("Source Text:", self.source_text)
        self.alternate_scorer = alternate_scorer



    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta).view(
                _B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        self.block_ngram_repeats(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        ## NOTE: custom code that reduces the probability of <unk> token to almose 0
        curr_scores[:,0] = -1e20

        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)
        if self.return_attention or self._cov_pen:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage = self._coverage.index_select(
                        1, self.select_indices)
                    self._coverage += current_attn
                    self._prev_penalty = self.global_scorer.cov_penalty(
                        self._coverage, beta=self.global_scorer.beta).view(
                            _B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(
                self._coverage,
                beta=self.global_scorer.beta)
            self.topk_scores -= cov_penalty.view(_B, self.beam_size)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                ## Only rerank with the LM probabilities if lm_model is given
                answer_not_in_hyp = False
                if self.lm_model or self.alternate_scorer:
                    word_hyp = [self._tgt_vocab.itos[id] if id < len(self._tgt_vocab.itos) else self._src_vocabs.itos[id - len(self._tgt_vocab.itos)] for id in predictions[i, j, 1:-1]]
                    if len(word_hyp) == 0:
                        continue
                    s = 0.0
                    if self.answer_text not in ' '.join(word_hyp) or "|||" in word_hyp:
                        # give a terrible score to this response
                        answer_not_in_hyp = True
                        s += -1e5
                    if self.lm_model:
                        # print("LM model")
                        str_hyp = ' '.join(word_hyp)
                        # Add the LM reranking score over here
                        lm_score = get_LM_prob(self.lm_model, str_hyp)
                        # print(i,j,predictions.shape)
                        # print("Hyp:", str_hyp)
                        # print("LM SCORE:", lm_score)
                        lm_score /= len(word_hyp)
                        # print("per word LM score:", lm_score)
                        # print()
                        # Update s here to include lm_score
                        score_demultiplier = self.LM_rerank_score_demultiplier
                        # print("HYP:", word_hyp)
                        # print(score_demultiplier * self.topk_scores[i, j] / len(word_hyp))
                        # print(lm_score)
                        s += score_demultiplier * self.topk_scores[i, j] / float(step + 1) + lm_score
                    if self.alternate_scorer:
                        # print("Alternate scorer")
                        s += float(self.global_scorer.alternate_model_score(self, word_hyp, self.topk_scores[i, j] / (step + 1))) 
                    s -= 0.075 * (step+1)
                    if self.lm_model and self.alternate_scorer:
                        # print("BOTH scorers")
                        # the main score has been counted twice. Remove the score_demultiplier part from LM model. We can manipulate the other two from alternate model scorer
                        s -= score_demultiplier * self.topk_scores[i, j] / float(step + 1)
                else:
                    # print(len(self._tgt_vocab.itos))
                    # print(self.source_text)
                    # print(len(self._src_vocabs.itos))
                    # print(predictions.shape)
                    # print(i)
                    # print(j)
                    # print([id for id in predictions[i, j, 1:-1]])
                    word_hyp = [self._tgt_vocab.itos[id] if id < len(self._tgt_vocab.itos) else self._src_vocabs.itos[id - len(self._tgt_vocab.itos)] for id in predictions[i, j, 1:-1]]
                    if len(word_hyp) == 0:
                        continue
                    s = 0.0
                    if self.answer_text not in ' '.join(word_hyp) or "|||" in word_hyp:
                        # give a terrible score to this response
                        answer_not_in_hyp = True
                        s += -1e5
                    else:
                        s = self.topk_scores[i, j] / (step + 1)

                if self.ratio > 0:
                    ## NOTE: if lm_model is present we will update the best score with lm_modified score
                    # s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s

                if not self.lm_model and not self.alternate_scorer:
                    flag = False
                    if not answer_not_in_hyp:
                        # if answer not in hyp then s is already -1e10
                        # So we will simply add the current score to it so that the ordering is still preserved after this penalty
                        s = 0.0
                    # else:
                    #     print(self.answer_text, "::", ' '.join(word_hyp), "::", s)
                    #     flag = True
                    s += self.topk_scores[i, j]
                    # if flag:
                    #     print("new s:", s, self.topk_scores[i, j], self.topk_scores[i, j]-1e5)

                
                
                self.hypotheses[b].append((
                    # self.topk_scores[i, j],
                    s,
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self._memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self._memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(step - 1, _B_new * self.beam_size, inp_seq_len)
            if self._cov_pen:
                self._coverage = self._coverage \
                    .view(1, _B_old, self.beam_size, inp_seq_len) \
                    .index_select(1, non_finished) \
                    .view(1, _B_new * self.beam_size, inp_seq_len)
                if self._stepwise_cov_pen:
                    self._prev_penalty = self._prev_penalty.index_select(
                        0, non_finished)

class AlternateModelScorer(object):
    """
    Final scorer which uses a different model to rerank the outputs
    """

    def __init__(self, model, fields, opts, final_prediction_score_multiper, alternate_model_score_multiplier, cuda=False):
        self.model = model
        self.fields = fields
        self.copy_attn = opts.copy_attn
        self.copy_attn = False
        self.src_field = self.fields["src"].base_field
        self.tgt_field = self.fields["tgt"].base_field
        self.src_vocab = self.fields["src"].base_field.vocab
        self.tgt_vocab = self.fields["tgt"].base_field.vocab
        self._tgt_pad_idx = self.tgt_vocab.stoi[self.tgt_field.pad_token]
        self.tt = torch.cuda if cuda else torch
        self.prediction_score_multiper = final_prediction_score_multiper
        self.alternate_model_score_multiplier = alternate_model_score_multiplier # S|T score multiplier
        # print("final_prediction_score_multiper:", self.prediction_score_multiper)
        # print("alternate_model_score_multiplier:", self.alternate_model_score_multiplier)
        # self.gamma = 0.0 # Length penalty multipler

        ## Some additional parameters from golbal scorer just to preserve consistency
        self.alpha = 0.0
        self.beta = 0.0
        penalty_builder = penalties.PenaltyBuilder('none', 'none')
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                warnings.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                warnings.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                warnings.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")

    def make_features(self, data):
        levels = [data]
        return torch.cat([level.unsqueeze(2) for level in levels], 2)

    def _run_target(self, src_data, hyp_data, src_lengths, src_map):
        src = Variable(self.make_features(src_data))
        # print("hyp_data shape:", hyp_data.size())
        # print(hyp_data)
        hyp_in = Variable(self.make_features(hyp_data)[:-1])
        # print(hyp_in.size())
        # print "src"
        # print src
        # print "src_lengths"
        # print src_lengths
        # print "hyp_in"
        # print hyp_in
        #  (1) run the encoder on the src
        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_state(src, memory_bank, enc_states)

        #  (2) Compute the 'goldScore'
        #  (i.e. log likelihood) of the source given target under the model (S|T)
        gold_scores = self.tt.FloatTensor(1).fill_(0)
        dec_out, dec_attn = self.model.decoder(
            # hyp_in, memory_bank, dec_states, memory_lengths=tgt_lengths)      #NOTE: For some unknown reasons it doesn't want memory_lenghts here
            hyp_in, memory_bank)

        ## This is the new way of calculating log prob with copy attn
        # print("COPY ATTN:", self.copy_attn)
        # print("DECODER out:", dec_out.size())
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()

        # print(log_probs)
        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = Variable(self.make_features(hyp_data)[1:])
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)
        # print("New score:", gold_scores)
        return gold_scores

    def tgt_to_index(self, tgt):
        return [self.tgt_vocab.stoi[word] for word in tgt]

    def src_to_index(self, src):
        return [self.src_vocab.stoi[word] for word in src]
    
    def alternate_model_score(self, beam, hyp_word, prediction_score):
        separator_index = beam.source_text.index("|||")
        src_text = ' '.join(beam.source_text[:separator_index])
        tgt_text = ' '.join(hyp_word)
        # print(src_text)
        # print(tgt_text)
        example = {"src":src_text, "tgt":tgt_text}
        src_ex_vocab, ex_dict = _dynamic_dict(example, self.src_field, self.tgt_field)
        src_map = ex_dict["src_map"]
        # print(src_map)
        # print(src_map.size())
        # exit()
        # Get the source
        src_list = self.src_to_index(beam.source_text) # beam's source is target here
        hyp_list = self.tgt_to_index(hyp_word)
        # Src list to torch LongTensor 
        hyp_list.insert(0, beam._bos)
        hyp_list.append(beam._eos)
        hyp_data = self.tt.LongTensor(hyp_list)
        src_lengths = self.tt.LongTensor([len(src_list) - 1])
        
        # print("src word:", src_word)
        # print("src_text:", beam.source_text)
        src = self.tt.LongTensor(src_list[:-1])         # Remove EOS from the src which is the src to the MMI model
        hyp_data.unsqueeze_(1)
        src.unsqueeze_(1)
        # print src
        score = self._run_target(src, hyp_data, src_lengths, src_map) / float(len(hyp_list))
        # print hyp_word
        # print score
        # print self.topic_score_multiplier * prediction_score, self.alternate_model_score_multiplier * float(score), self.gamma * len(hyp_word)
        # print hyp_word
        # print ""
        return self.prediction_score_multiper * prediction_score + self.alternate_model_score_multiplier * score / float(len(hyp_list)-1)

    ##NOTE: Blatantly copied from other class. Don't know if this will work
    def update_score(self, beam, attn):
        """Update scores of a Beam that is not finished."""
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        """Keeps the coverage vector as sum of attentions."""
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
