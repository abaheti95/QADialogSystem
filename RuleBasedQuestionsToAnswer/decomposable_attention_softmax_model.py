from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, sequence_cross_entropy_with_logits
# from allennlp.training.metrics import CategoricalAccuracy
from categorical_accuracy_for_softmax import CategoricalAccuracySoftmax


@Model.register("decomposable_attention_softmax")
class DecomposableAttentionSoftmax(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.
    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 loss_type: str = '_nll') -> None:
        super(DecomposableAttentionSoftmax, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
                               "text field embedding dim", "attend feedforward input dim")
        # We are modifying this because our number of labels are 2 now and we want to get a softmax on the final logits
        # check_dimensions_match(aggregate_feedforward.get_output_dim(), self._num_labels,
        #                        "final output dimension", "number of labels")

        self._accuracy = CategoricalAccuracySoftmax()
        self._loss = torch.nn.CrossEntropyLoss()
        self._loss_type=loss_type

        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypotheses: Dict[str, torch.LongTensor],
                labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional, (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # print(hypotheses.keys())
        # print("Hypotheses")
        # print(type(hypotheses["tokens"]))
        # print("Hypotheses:", hypotheses["tokens"].shape)
        # print("Premise")
        # print(type(premise["tokens"]))
        # print("Premise:", premise["tokens"].shape)
        # print("Metadata")
        # print(type(metadata[0]))
        # print(len(metadata[0]['labels']))
        embedded_premise = self._text_field_embedder(premise)
        premise_mask = get_text_field_mask(premise).float()
        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise, premise_mask)
        projected_premise = self._attend_feedforward(embedded_premise)
        # print("Projected premise", projected_premise.shape)


        all_label_logits = list()
        all_label_probs = list()
        all_h2p_attention = list()
        all_p2h_attention = list()
        embedded_hypotheses = self._text_field_embedder(hypotheses)
        hypotheses_mask = get_text_field_mask(hypotheses, num_wrapping_dims=1).float()
        projected_hypotheses = self._attend_feedforward(embedded_hypotheses)
        # print("Projected Hypotheses", projected_hypotheses.shape)
        # print("Embedded premise:", embedded_premise.shape)
        # print("Embedded Hypotheses:", embedded_hypotheses.shape)
        premise_length = projected_premise.shape[1]
        num_hypotheses = projected_hypotheses.shape[1]
        hypotheses_length = projected_hypotheses.shape[2]
        batch_size = projected_hypotheses.shape[0]
        projection_size = projected_hypotheses.shape[3]
        # reshape the hypotheses to: (batch_size, num_hypotheses * hypotheses_length, projection_size)
        projected_hypotheses_squashed = projected_hypotheses.view(batch_size, num_hypotheses*hypotheses_length, projection_size)
        # print("projected_hypotheses_squashed", projected_hypotheses_squashed.shape)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix_squashed = self._matrix_attention(projected_premise, projected_hypotheses_squashed)
        similarity_matrix = similarity_matrix_squashed.view(batch_size, premise_length, num_hypotheses, hypotheses_length)
        # print("Similarity Matrix", similarity_matrix.shape)
        
        # Shape: (batch_size, premise_length, num_hypotheses, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypotheses_mask)
        p2h_attention_summed = torch.sum(p2h_attention, dim=3)
        # print("p2h_attention_summed", p2h_attention_summed.shape)
        # print(p2h_attention_summed)
        # print(p2h_attention[0])

        all_p2h_attention.append(p2h_attention)

        # print("p2h attention", p2h_attention.shape)
        # Shape: (batch_size, premise_length, num_hypotheses, embedding_dim)
        attended_hypotheses = weighted_sum(embedded_hypotheses, p2h_attention)

        # print("Premise Length", premise_length)
        # print("Num Hypotheses", num_hypotheses)
        # print("Hypotheses Length", hypotheses_length)
        # print("Projection size", projection_size)

        # print("Similarity Matrix Squashed", similarity_matrix_squashed.shape)
        # print("Similarity Matrix Squashed Transposed", similarity_matrix_squashed.transpose(1, 2).contiguous().shape)
        # Shape: (batch_size, num_hypotheses*hypotheses_length, premise_length)
        h2p_attention_squashed = masked_softmax(similarity_matrix_squashed.transpose(1, 2).contiguous(), premise_mask)
        # print("H2P attention Squashed", h2p_attention_squashed.shape)
        # Shape: (batch_size, num_hypotheses, hypotheses_length, premise_length)
        h2p_attention = h2p_attention_squashed.view(batch_size, num_hypotheses, hypotheses_length, premise_length)
        all_h2p_attention.append(h2p_attention)

        # Shape: (batch_size, hypothesis_length, embedding_dim)
        # print("H2P attention", h2p_attention.shape)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        # Repeat embedded_premise * num_hypotheses times
        num_hypotheses_embedded_premise = torch.stack([embedded_premise]*num_hypotheses).transpose(0,1).contiguous()
        # print("Embedded premise", embedded_premise.shape)
        # print("Repeated embedded_premise:", num_hypotheses_embedded_premise.shape)

        # print("Attended Hypotheses", attended_hypotheses.shape)
        attended_hypotheses_transposed = attended_hypotheses.transpose(1, 2).contiguous()
        # print("Attented Hypotheses Transposed", attended_hypotheses_transposed.shape)
        premise_compare_input = torch.cat([num_hypotheses_embedded_premise, attended_hypotheses_transposed], dim=-1)

        # print("Embedded Hypotheses", embedded_hypotheses.shape)
        # print("Attended premise:", attended_premise.shape)
        hypotheses_compare_input = torch.cat([embedded_hypotheses, attended_premise], dim=-1)
        

        # print("Premise Compare Input", premise_compare_input.shape)
        compared_premise = self._compare_feedforward(premise_compare_input)
        # print("Compared premise", compared_premise.shape)
        # print("Premise mask", premise_mask.shape)
        # print("Premise mask unsqueezed", premise_mask.unsqueeze(-1).shape)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # print("Compared Premise after masking", compared_premise.shape)
        # Shape: (batch_size, num_hypotheses, compare_dim)
        compared_premise = compared_premise.sum(dim=2)
        # print("Summed compared premise", compared_premise.shape)


        # print("Hypotheses compare input", hypotheses_compare_input.shape)
        compared_hypotheses = self._compare_feedforward(hypotheses_compare_input)
        # print("Compare hypotheses", compared_hypotheses.shape)
        # print("Hypotheses mask", hypotheses_mask.shape)
        # print("Hypotheses mask unsqueezed", hypotheses_mask.unsqueeze(-1).shape)
        compared_hypotheses = compared_hypotheses * hypotheses_mask.unsqueeze(-1)
        # print("Compared Hypotheses after masking", compared_hypotheses.shape)
        # Shape: (batch_size, num_hypotheses, compare_dim)
        compared_hypotheses = compared_hypotheses.sum(dim=2)
        # print("Summed compared hypotheses", compared_hypotheses.shape)

        aggregate_input = torch.cat([compared_premise, compared_hypotheses], dim=-1)
        # print("Aggregate Input", aggregate_input.shape)
        label_logits = self._aggregate_feedforward(aggregate_input)
        # print("label_logits", label_logits.shape)
        label_logits_squeezed = label_logits.squeeze(-1)
        # print("label logit squeezed", label_logits_squeezed.shape)
        all_label_logits.append(label_logits_squeezed)
        # How to apply softmax on all_label_logits
        label_probs = torch.nn.functional.softmax(label_logits_squeezed, dim=-1)
        log_label_probs = -torch.nn.functional.log_softmax(label_logits_squeezed, dim=-1)
        # print("label_probs", label_probs.shape)
        output_dict = {"label_logits": all_label_logits,
                       "label_probs": label_probs,
                       "h2p_attention": all_h2p_attention,
                       "p2h_attention": all_p2h_attention}

        # How to compute correct loss here?
        if labels is not None:
            # We only care about the values which are 1 in the labels
            # print()
            # print("Label logits squeezed", label_logits_squeezed)
            # print("Label probs", label_probs)
            # log_label_probs = -torch.log(label_probs)
            # print("Log Label Probs", log_label_probs)
            # print("labels type", type(labels))
            # print("labels", labels)
            # NLL loss
            if self._loss_type == "_nll":
                loss = log_label_probs * labels.float()
                # print("Loss", loss.shape)
                # print("Loss", loss)
                loss = torch.sum(loss)
            # MSE loss
            elif self._loss_type == "_mse":
                mse = torch.nn.MSELoss()
                loss = mse(label_probs, labels.float())
            else:
                print("ERROR: unkown loss type:", self._loss_type)
                exit(1)
            # print("final loss", loss)
            # print()
            self._accuracy(label_logits_squeezed, labels)
            output_dict["loss"] = loss
            
        # if metadata is not None:
        #     output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
        #     output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }