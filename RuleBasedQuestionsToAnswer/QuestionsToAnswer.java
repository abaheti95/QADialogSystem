import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

// import edu.cmu.ark.AnalysisUtilities;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.tregex.TregexMatcher;
import edu.stanford.nlp.trees.tregex.TregexPattern;
import edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon;
import edu.stanford.nlp.trees.tregex.tsurgeon.TsurgeonPattern;

// Useful for pos tagging
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

// Importing SimpleNLG to convert verb tense
import simplenlg.framework.*;
import simplenlg.lexicon.*;
import simplenlg.realiser.english.*;
import simplenlg.phrasespec.*;
import simplenlg.features.*;

class Pair {
	public String response;
	public String rule;

	public Pair() {
		this.response = "";
		this.rule = "";
	}

	public Pair(String response, String rule) {
		this.response = response;
		this.rule = rule;
	}
}

public class QuestionsToAnswer{
	public static Realiser realiser;
	public static String DB_FILENAME = "lexAccess2016lite/data/HSqlDb/lexAccess2016.data";
	public static Lexicon lexicon;
	public static String[] all_wh_questions = {"what", "who", "whom", "whose", "when", "where", "which", "why", "how"};
	public static String[] all_prepositions = {"in", "on", "at", "by", "as", "of", "for", "from", "about", "to", "with", "because", ""};
	// TODO: debugging with lesser prepositions to see if size is the issue?
	// public static String[] all_prepositions = {"in", "on", "at", "by", "as", "of", "for", "about", "to", "with", "because", ""};
	public static String[] all_dets = {"a", "an", "the"};
	public static String[] all_pronouns = {"it", "they", "he", "she"};
	public static String[] all_special_cases = {"an example of", "the example of", "a name of", "the name of"};
	

	public static String change_word_tense(String word_str, String tense) {
		WordElement word = lexicon.getWord(word_str, LexicalCategory.VERB);
		InflectedWordElement inflWord = new InflectedWordElement(word);
		
		if(tense.equals("VBD")) {
			inflWord.setFeature(Feature.TENSE, Tense.PAST);
		} else if(tense.equals("VBZ")) {
			// inflWord.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
			inflWord.setFeature(Feature.TENSE, Tense.PRESENT);
			inflWord.setFeature(Feature.PERSON, Person.THIRD);
		} else if(tense.equals("VBP")) {
			inflWord.setFeature(Feature.PERSON, Person.FIRST);
			inflWord.setFeature(Feature.TENSE, Tense.PRESENT);
		}
		String changed_word = realiser.realise(inflWord).toString();
		if(changed_word.endsWith("eded")) {
			changed_word = changed_word.substring(0, changed_word.lastIndexOf("ed"));
		}
		return changed_word;
	}

	public static String tree_to_leaves_string(Tree t) {
		if(t==null)
			return "";
		StringBuilder sb = new StringBuilder();
		for (Tree s : t.getLeaves()) {
			sb.append(s.value());
			sb.append(" ");
		}
		return sb.toString().trim();
	}

	// Consider adding ADJP
	public static String get_prep_from_vp(Tree vp) {
		// See if there is a PP just after VP which has the IN as its first child and optionally PP as its second child
		Tree current = vp;
		String preposition = "";
		if(vp == null) {
			// When the input is null just return an empty list
			return "";
		}
		if(current.lastChild() != null) {
			current = current.lastChild();
			System.out.println(current + " : " + current.firstChild());
			if(current.value().equals("PP")) {
				// Check if the left child is IN and has only one word and the right child is a PP.. Only in this case the child is a correct Prep
				if((current.firstChild().value().equals("IN") || current.firstChild().value().equals("TO")) && current.firstChild().children().length==1 && (current.lastChild().nodeString().equals("PP"))) {
					// do not add this node as this is just an important preposition before a preposition phrase or a subordinate clause
					preposition = tree_to_leaves_string(current.firstChild());
				} else if(current.children().length == 1 && (current.firstChild().nodeString().equals("IN") || current.firstChild().nodeString().equals("TO"))) {
					// if current has only one child and it is a IN or TO then too its an important preposition and should be returned
					preposition = tree_to_leaves_string(current.firstChild());
				} else {
					// do nothing.. no preposition found
				}
			}
		}
		return preposition;
	}
	public static String replaceLast(String string, String toReplace, String replacement) {
		int pos = string.lastIndexOf(toReplace);
		if (pos > -1) {
			return string.substring(0, pos)
			+ replacement
			+ string.substring(pos + toReplace.length(), string.length());
		} else {
			return string;
		}
	}

	public static String get_vp_without_prep(Tree vp, String preposition) {
		Tree current = vp;
		String full_vp = " " + tree_to_leaves_string(current) + " ";
		// Prep is at current.lastChild().firstChild()
		String prep_phrase_in_vp = " " + tree_to_leaves_string(current.lastChild()) + " ";
		String prep_phrase_in_vp_without_preposition = " " + prep_phrase_in_vp.replaceFirst(Pattern.quote(" " + preposition + " "), " ").trim() + " ";
		System.out.println(full_vp);
		System.out.println(prep_phrase_in_vp);
		System.out.println(prep_phrase_in_vp_without_preposition);

		return replaceLast(full_vp, prep_phrase_in_vp, prep_phrase_in_vp_without_preposition).trim();
	}

	public static ArrayList<Tree> get_nested_pp_list(Tree vp) {
		// Iterate over last child and see if there are nested PP
		Tree current = vp;
		ArrayList<Tree> nested_pp = new ArrayList<Tree>();
		if(vp == null) {
			// When the input is null just return an empty list
			return nested_pp;
		}
		while(current.lastChild() != null) {
			current = current.lastChild();
			if(current.value().equals("PP") || current.value().equals("SBAR") || current.value().equals("ADJP")) {
				// Check if the left child is IN and has only one word and the right child is a PP or SBAR.. Then don't add this in the list
				if((current.firstChild().value().equals("IN") || current.firstChild().value().equals("TO")) && current.firstChild().children().length==1 && (current.lastChild().nodeString().equals("PP") || current.lastChild().nodeString().equals("SBAR") || current.lastChild().nodeString().equals("ADJP"))) {
					// do not add this node as this is just an important preposition before a preposition phrase or a subordinate clause
				} else if(current.children().length == 1 && (current.firstChild().nodeString().equals("IN") || current.firstChild().nodeString().equals("TO"))) {
					// if current has only one child and it is a IN or TO then too its an important preposition and should not be added to this list
				} else {
					nested_pp.add(current);
				}
			}
		}
		return nested_pp;
	}
	
	public static boolean check_if_np_is_pronoun(Tree np) {
		if(np == null)
			return false;
		// If np is already a pronoun then we need to ignore it
		// if the leftmost child is a pronoun then we need to ignore it
		// System.out.println("NP:" + np + " : " + np.firstChild().value());
		if(np.firstChild().value().equals("PRP") || np.firstChild().value().equals("PRP$")) {
			if(np.children().length > 1) {
				// System.out.println("YE NP:" + np);
			}
			return true;
		}
		return false;
	}

	public static ArrayList<String> get_string_from_tree_list(ArrayList<Tree> tlist) {
		ArrayList<String> slist = new ArrayList<String>();
		for(Tree t : tlist) {
			slist.add(tree_to_leaves_string(t));
		}
		return slist;
	}

	public static HashMap<String, Tree> search_pattern(String pattern, Tree tree, ArrayList<String> variable_names) {
		// We will create a common function which will search the provided pattern on the tree and return all the variables with their nodes in a HashMap
		HashMap<String, Tree> found_variables = new HashMap<String, Tree>();

		// Create the pattern and matcher
		TregexPattern tpattern = TregexPattern.compile(pattern);
		TregexMatcher matcher = tpattern.matcher(tree);
		try {
			if(matcher.find()) {
				for(String variable_name : variable_names) {
					found_variables.put(variable_name, matcher.getNode(variable_name));
				}
			}
		} catch(NullPointerException e) {
			// do nothing as the expression could not be found
			System.out.println("NULL found here!!");
		}
		return found_variables;
	}

	// Function to remove duplicates from an ArrayList 
	public static ArrayList<Pair> remove_duplicate_responses(ArrayList<Pair> responses_and_rules) {
		// Create a new LinkedHashSet 
		ArrayList<Pair> unique_responses_and_rules = new ArrayList<Pair>();
		HashSet<String> unique_responses = new HashSet<String>();

		for(Pair response_and_rule: responses_and_rules) {
			// trim all the responses first
			String response = response_and_rule.response.replaceAll("\\s+", " ").trim();
			String rule = response_and_rule.rule;
			if(!unique_responses.contains(response)) {
				unique_responses_and_rules.add(response_and_rule);
				unique_responses.add(response);
			}
		}
		// return the list 
		return unique_responses_and_rules;
	}
	
	public static void print_all_responses(ArrayList<Pair> responses_and_rules) {
		for(int i = 0; i < responses_and_rules.size(); i ++) {
			System.out.println("Rule" + (i+1) + ":" + responses_and_rules.get(i).rule);
			System.out.println("Answer" + (i+1) + ":" + responses_and_rules.get(i).response);
		}
	}
	
	public static Tree remove_only_child(Tree question) {
		// If SQ has only one child which is VP then excise it
		TregexPattern question_pattern = TregexPattern.compile("SBARQ < (SQ=main_clause <: VP=only_child)");
		// TregexPattern question_pattern = TregexPattern.compile("SBARQ < (SQ=main_clause < VP=only_child)");
		Tree mod_question = question.deepCopy();
		List<TsurgeonPattern> ps = new ArrayList<TsurgeonPattern>();
		ps.add(Tsurgeon.parseOperation("excise only_child only_child"));
		Tsurgeon.processPattern(question_pattern, Tsurgeon.collectOperations(ps), mod_question);
		question = mod_question;

		question_pattern = TregexPattern.compile("SBARQ << (S=only_parent <: /VP|NP/)");
		// TregexPattern question_pattern = TregexPattern.compile("SBARQ < (SQ=main_clause < VP=only_child)");
		mod_question = question.deepCopy();
		ps = new ArrayList<TsurgeonPattern>();
		ps.add(Tsurgeon.parseOperation("excise only_parent only_parent"));
		Tsurgeon.processPattern(question_pattern, Tsurgeon.collectOperations(ps), mod_question);
		question = mod_question;

		//in what forms of media has frédéric been the subject of ?
		// question_pattern = TregexPattern.compile("SBARQ < (/WH.?/=wh_phrase << (S=only_parent <: /VP|NP/))");
		// // TregexPattern question_pattern = TregexPattern.compile("SBARQ < (SQ=main_clause < VP=only_child)");
		// mod_question = question.deepCopy();
		// ps = new ArrayList<TsurgeonPattern>();
		// ps.add(Tsurgeon.parseOperation("excise only_parent only_parent"));
		// Tsurgeon.processPattern(question_pattern, Tsurgeon.collectOperations(ps), mod_question);
		return mod_question;
	}

	public static ArrayList<Pair> wh_did_question_to_answer(Tree wh_did_question, String answer, String wh_word) {
		ArrayList<Pair> gen_responses_and_rules = new ArrayList<Pair>();
		HashMap<String, Tree> found_variables;
		HashMap<String, Object> required_terms = new HashMap<String, Object>();
		String rules = "R_1 ";			// We will store the sequence code path or used rules in this string. Will be later used as a feature in the models 

		found_variables = search_pattern("SBARQ < (SQ < /VB.?|MD/=tense_verb < ((VP=verb_phrase ?$-- ADVP=adverb) < (/@VP|VB.?/=main_verb ?$-- ADVP=adverb2)))", wh_did_question, new ArrayList<String>(Arrays.asList("tense_verb", "main_verb", "verb_phrase", "adverb", "adverb2")));
		if(found_variables.isEmpty()) {
			System.out.println("ERROR in Parse Tree:" + wh_did_question);
		} else {
			String tense = "", tense_verb = "", main_verb = "", adverb_main_verb1 = "", adverb_main_verb2 = "", changed_verb = "", verb_phrase = "", np_str = "", prep_phrase = "";
			ArrayList<String> prep_phrases_in_vp, prep_phrases_in_np, prep_phrases_in_np_in_wh;
			// System.out.println("Main verb initially:" + found_variables.get("main_verb"));
			// Sometimes the main verb will be VP then its first child might be VB
			if(found_variables.get("main_verb").value().equals("VP")) {
				if(found_variables.get("main_verb").firstChild().value().startsWith("VB")) {
					found_variables.put("main_verb", found_variables.get("main_verb").firstChild());
				} else if(found_variables.get("main_verb").value().equals("VP")){
					System.out.println("ERROR2 in Parse Tree:" + wh_did_question);
					return gen_responses_and_rules;
				}
			}

			// Get the tense verb, adverb, main verb and verb phrase
			tense = found_variables.get("tense_verb").nodeString();
			tense_verb = tree_to_leaves_string(found_variables.get("tense_verb"));


			// System.out.println("TENSE VERB: " + tense_verb);
			verb_phrase = tree_to_leaves_string(found_variables.get("verb_phrase"));
			// System.out.println("Verb Phrase:" + verb_phrase);
			
			// Get the nested PP inside the Verb Phrase
			Tree vp_node = found_variables.get("verb_phrase");

			adverb_main_verb1 = tree_to_leaves_string(found_variables.get("adverb"));
			adverb_main_verb2 = tree_to_leaves_string(found_variables.get("adverb2"));

			// Change the tense of the main verb based on the tense verb
			main_verb = found_variables.get("main_verb").firstChild().value();
			// NOTE: Handle the special cases when "will", "would" or "can" are the main verbs. In this case we don't need to change the tense. Keep the verb phrase as it is and add the tense verb to it as well
			if(tense_verb.equals("will") || tense_verb.equals("would") || tense_verb.equals("can") || tense_verb.equals("must") || tense_verb.equals("may") || tense_verb.equals("should") || tense_verb.equals("could")) {
				rules += "R_will ";			// The tense verb can be (will, would, can, must, may, should, could)
				changed_verb = tense_verb + " " + adverb_main_verb1 + ((!adverb_main_verb1.isEmpty() && !adverb_main_verb2.isEmpty())? " ": "") + adverb_main_verb2 +  " " + main_verb;
				changed_verb = changed_verb.replaceAll("\\s+", " ");
				// System.out.println("SPECIAL CHANGES");
			} else {
				rules += "R_did ";			// The tense verb can be (do, does, did)
				changed_verb = adverb_main_verb1 + adverb_main_verb2 + " " + change_word_tense(main_verb, tense);
				if(!((adverb_main_verb1 + adverb_main_verb2).trim()).isEmpty()) {
					// System.out.println("SPECIAL CHANGES 2: " + changed_verb);
				}
			}

			if(!adverb_main_verb1.isEmpty()) {
				// System.out.println("Adverb1:" + adverb_main_verb1);
				rules += "R_adv1 ";
			}
			if(!adverb_main_verb2.isEmpty()) {
				// System.out.println("Adverb2:" + adverb_main_verb2);
				rules += "R_adv2 ";
			}

			// System.out.println("Main verb:" + main_verb);
			main_verb = adverb_main_verb1 + ((!adverb_main_verb1.isEmpty() && !adverb_main_verb2.isEmpty())? " ": "") + adverb_main_verb2 + " " + main_verb;
			main_verb = main_verb.trim();
			// System.out.println("Main verb:" + main_verb);
			// System.out.println("Changed Verb:" + changed_verb);
			// Adverb1 is outside verb phrase therefore we need to add it ourselves
			verb_phrase = adverb_main_verb1 + " " + verb_phrase;

			// Get the NP between tense_verb and main_verb if any
			found_variables = search_pattern("SBARQ < (SQ < (/VB.?|MD/ $++ (/NP|RB/=np $++ VP)))", wh_did_question, new ArrayList<String>(Arrays.asList("np")));
			np_str = tree_to_leaves_string(found_variables.get("np"));
			// if(np_str.startsWith("you")) {
			// 	System.out.println("YOU yes YOU!");
			// }
			boolean np_is_pronoun = check_if_np_is_pronoun(found_variables.get("np"));
			if(np_is_pronoun) {
				rules += "R_np_is_prp ";
			}
			// Get the nested PP inside the Noun Phrase
			prep_phrases_in_np = get_string_from_tree_list(get_nested_pp_list(found_variables.get("np")));
			if(prep_phrases_in_np.size() > 0) {
				rules += "R_pp_in_np ";
			}
			// System.out.println("NP STR:" + found_variables.get("np"));
			// System.out.println("PPs in NP:" + prep_phrases_in_np);
			// Add a empty string element so that all the answers can be generated in one loop
			prep_phrases_in_np.add("");

			// Get the subordinate clause if present
			found_variables = search_pattern("SBARQ < (SBAR=subordinate)", wh_did_question, new ArrayList<String>(Arrays.asList("subordinate")));
			String subordinate_clause = "";
			subordinate_clause = tree_to_leaves_string(found_variables.get("subordinate"));
			if(!subordinate_clause.isEmpty()) {
				rules += "R_subordinate_clause ";
			}

			// Get prepositions before the wh-phrase
			found_variables = search_pattern("SBARQ=question_head < (/WH.?/=wh_phrase << " + wh_word + " ?< /NP|NN/=np_in_wh_phrase)", wh_did_question, new ArrayList<String>(Arrays.asList("question_head", "wh_phrase", "np_in_wh_phrase")));
			String preposition = "";
			String wh_phrase = tree_to_leaves_string(found_variables.get("wh_phrase"));
			String np_in_wh_phrase = tree_to_leaves_string(found_variables.get("np_in_wh_phrase")).replace(wh_word, "").trim();
			prep_phrases_in_np_in_wh = get_string_from_tree_list(get_nested_pp_list(found_variables.get("np_in_wh_phrase")));
			if(prep_phrases_in_np_in_wh.size() > 0) {
				rules += "R_pp_in_np_in_wh ";
			}
			if(!np_in_wh_phrase.isEmpty()) {
				// System.out.println("YEAH found NP in WH:" + np_in_wh_phrase);
				rules += "R_np_in_wh ";
				prep_phrases_in_np_in_wh.add(0, np_in_wh_phrase);
			}
			prep_phrases_in_np_in_wh.add("");
			preposition = wh_phrase.substring(0, wh_phrase.indexOf(wh_word)).trim();
			if(!preposition.isEmpty()) {
				rules += "R_prep_in_wh ";
			}

			prep_phrases_in_vp = get_string_from_tree_list(get_nested_pp_list(vp_node));
			if(prep_phrases_in_vp.size() > 0) {
				rules += "R_pp_in_vp ";
			}
			// System.out.println("PPs in VP:" + prep_phrases_in_vp);
			// Add a empty string element so that all the answers can be generated in one loop
			prep_phrases_in_vp.add("");

			if(preposition.isEmpty()) {
				// NOTE: probably because in why question is not a good idea here
				// preposition = (wh_word.equals("why"))? "because {missing_IN}" : "{missing_IN}";
				// check for the preposition in vp
				preposition = get_prep_from_vp(vp_node);
				// need to remove this prep from final_verb and its pps
				if(!preposition.isEmpty()) {
					rules += "R_prep_in_vp ";
					verb_phrase = adverb_main_verb1.trim() + " " +  get_vp_without_prep(vp_node, preposition);
					System.out.println("VERB PHRASE:" + verb_phrase + ": vs :" + adverb_main_verb1);
				}
				for(int i = 0; i < prep_phrases_in_vp.size(); i++) {
					String pp = " " + prep_phrases_in_vp.get(i) + " ";
					prep_phrases_in_vp.set(i, pp.replaceFirst(Pattern.quote(" " + preposition + " "), " ").trim());
				}
				if(preposition.isEmpty()) {
					rules += "R_missing_prep ";
					preposition = "{missing_IN}";
				}
			}

			verb_phrase = " " + verb_phrase.trim() + " ";
			// System.out.println("Main verb:" + main_verb);
			// System.out.println("verb phrase:" + verb_phrase);
			// System.out.println("Changed verb:" + changed_verb);
			String response = "", verb_phrase1 = "", verb_phrase2 = "", verb_phrase3 = "";
			if(!verb_phrase.contains(main_verb)) {
				rules += "R_ERROR_no_mainv_in_vp ";
				System.out.println("SERIOUS Error:" + verb_phrase + ": vs :" + main_verb + ":");
				return gen_responses_and_rules;
			}
			for(String pp_in_np_in_wh : prep_phrases_in_np_in_wh) {
				String answer_with_brackets = "{" + answer + "}" + (!np_in_wh_phrase.isEmpty()?(" " + np_in_wh_phrase.replace(pp_in_np_in_wh, "").trim()):"");
				String pre_current_rules = rules;
				if(!np_in_wh_phrase.isEmpty() && np_in_wh_phrase.replace(pp_in_np_in_wh, "").trim().isEmpty()) {
					// np_in_wh_phrase is not added to the answer phrase
				} else if(!np_in_wh_phrase.isEmpty()){
					// np_in_wh_phrase is added to the answer phrase
					pre_current_rules += "R_np_in_wh_added ";
				}
				for(int i = 0; i < ((subordinate_clause.isEmpty()) ? 1 : 3); i++) {
					String current_rules = pre_current_rules;
					if(i == 1) {
						current_rules += "R_sub_first ";
						// This means subordinate clause is present so it should be before the response
						response = subordinate_clause + " , []";

					} else if (i == 2) {
						current_rules += "R_sub_last ";
						// This means subordinate clause is present and should be after the response
						response = "[] , " + subordinate_clause;
					} else {
						current_rules += "R_no_sub ";
						response = "[]";
					}
					// keep answer in the beginning
					verb_phrase1 = verb_phrase.replace(" " + main_verb + " ", " " + changed_verb + " ");
					for(String pp_vp : prep_phrases_in_vp) {
						String pre_final_rules = current_rules + "R_ans_first ";
						if(!verb_phrase1.contains(pp_vp)) {
							pre_final_rules += "R_ERROR_no_pp_vp_found_in_vp1 ";
							System.out.println("BADUM:" + verb_phrase1);
							System.out.println("BADUM:" + verb_phrase + ":");
							System.out.println("BADUM:" + main_verb + ":");
							System.out.println("BADUM:" + changed_verb + ":");
							System.out.println("BADUM:" + pp_vp);
						}
						if(pp_vp.isEmpty()) {
							pre_final_rules += "R_pp_in_vp_not_removed ";
						} else {
							// removing pp in vp so add the rule
							pre_final_rules += "R_pp_in_vp_removed ";
						}
						for(String pp_np : prep_phrases_in_np) {
							String final_rules = pre_final_rules;
							if(pp_np.isEmpty()) {
								final_rules += "R_pp_in_np_not_removed ";
							} else {
								// removing pp so add the rule
								final_rules += "R_pp_in_np_remvoed ";
							}
							gen_responses_and_rules.add(new Pair(response.replace("[]", preposition + " " + answer_with_brackets + " , " + np_str.replace(pp_np, "") + " " + verb_phrase1.replace(pp_vp, "")).replaceAll("\\s+", " "), final_rules.trim()));
						}
						// replace the np_str with pronouns
						if(!np_str.isEmpty() && !np_is_pronoun) {
							for(String pronoun : all_pronouns) {
								String final_rules = pre_final_rules + "R_np_replaced_with_pronoun ";
								gen_responses_and_rules.add(new Pair(response.replace("[]", preposition + " " + answer_with_brackets + " , " + pronoun + " " + verb_phrase1.replace(pp_vp, "")).replaceAll("\\s+", " "), final_rules.trim()));
							}
						}
					}

					// Attach the answer just after the changed verb phrase with preposition
					verb_phrase2 = verb_phrase.replace(" " + main_verb + " ", " " + changed_verb + " ") + " " + preposition + " " + answer_with_brackets + " ";
					for(String pp_vp : prep_phrases_in_vp) {
						String pre_final_rules = current_rules + "R_ans_after_vp ";
						if(!verb_phrase2.contains(pp_vp)) {
							pre_final_rules += "R_ERROR_no_pp_vp_found_in_vp2 ";
							System.out.println("BADUM:" + verb_phrase2);
							System.out.println("BADUM:" + pp_vp);
						}
						if(pp_vp.isEmpty()) {
							pre_final_rules += "R_pp_in_vp_not_removed ";
						} else {
							// removing pp in vp so add the rule
							pre_final_rules += "R_pp_in_vp_removed ";
						}
						for(String pp_np : prep_phrases_in_np) {
							String final_rules = pre_final_rules;
							if(pp_np.isEmpty()) {
								final_rules += "R_pp_in_np_not_removed ";
							} else {
								// removing pp so add the rule
								final_rules += "R_pp_in_np_remvoed ";
							}
							gen_responses_and_rules.add(new Pair(response.replace("[]", np_str.replace(pp_np, "") + " " + verb_phrase2.replace(pp_vp, "")).replaceAll("\\s+", " "), final_rules.trim()));
						}
						// replace the np_str with pronouns
						if(!np_str.isEmpty() && !np_is_pronoun) {
							for(String pronoun : all_pronouns) {
								String final_rules = pre_final_rules + "R_np_replaced_with_pronoun ";
								gen_responses_and_rules.add(new Pair(response.replace("[]", pronoun + " " + verb_phrase2.replace(pp_vp, "")).replaceAll("\\s+", " "), final_rules.trim()));
							}	
						}
					}

					// Attach the answer just after the changed verb with preposition so that the PP comes after the answer
					// Only required if there is something after main verb in vp
					if(!verb_phrase.trim().endsWith(main_verb.trim())) {
						verb_phrase3 = verb_phrase.replace(" " + main_verb + " ", " " + changed_verb + " " + preposition + " " + answer_with_brackets + " ");
						for(String pp_vp : prep_phrases_in_vp) {
							String pre_final_rules = current_rules + "R_ans_after_verb ";
							if(pp_vp.isEmpty()) {
								pre_final_rules += "R_pp_in_vp_not_removed ";
							} else {
								// removing pp in vp so add the rule
								pre_final_rules += "R_pp_in_vp_removed ";
							}
							if(!verb_phrase3.contains(pp_vp)) {
								pre_final_rules += "R_ERROR_no_pp_vp_found_in_vp3 ";
								System.out.println("BADUM:" + verb_phrase3);
								System.out.println("BADUM:" + pp_vp);
							}
							for(String pp_np : prep_phrases_in_np) {
								String final_rules = pre_final_rules;
								if(pp_np.isEmpty()) {
									final_rules += "R_pp_in_np_not_removed ";
								} else {
									// removing pp so add the rule
									final_rules += "R_pp_in_np_remvoed ";
								}
								gen_responses_and_rules.add(new Pair(response.replace("[]", np_str.replace(pp_np, "") + " " + verb_phrase3.replace(pp_vp, "")).replaceAll("\\s+", " "), final_rules.trim()));
							}
							// replace the np_str with pronouns
							if(!np_str.isEmpty() && !np_is_pronoun) {
								for(String pronoun : all_pronouns) {
									String final_rules = pre_final_rules + "R_np_replaced_with_pronoun ";
									gen_responses_and_rules.add(new Pair(response.replace("[]", pronoun + " " + verb_phrase3.replace(pp_vp, "")).replaceAll("\\s+", " "), final_rules.trim()));
								}
							}
						}
					}
				}
			}
		}
		return gen_responses_and_rules;
	}

	// TODO: Continue here!
	public static ArrayList<Pair> wh_is_question_to_answer(Tree wh_is_question, String answer, String wh_word) {
		/* Find the main verb and the second verb in the parse tree. Extract the NP in between them. Answer: NP + main verb + final verb + remaining final verb phrase + prep + {answer}
		Prep + {answer} + NP + main verb + final verb + remaining final vp */

		ArrayList<Pair> gen_responses_and_rules = new ArrayList<Pair>();
		HashMap<String, Tree> found_variables;

		// Get everything after is except and identify the last vp is present
		found_variables = search_pattern("SBARQ < (SQ=clause < (/VB.?/=verb ?$-- ADVP=adverb1 ?$++ NP=np) ?<- (/VP|NP/=final_verb ?$-- ADVP=adverb2))", wh_is_question, new ArrayList<String>(Arrays.asList("clause", "verb", "np", "adverb1", "final_verb", "adverb2")));
		String rules = "R_2 ";			// We will store the sequence code path or used rules in this string. Will be later used as a feature in the models 
		
		System.out.println("WH IS QUESTION");
		if(!found_variables.isEmpty()) {
			// Get everything after is except and identify the last vp is present
			String np_str = "", clause_str = "" , is_verb_str = "", adverb1_str="", adverb2_str="", final_verb_phrase_str = "";
			Tree final_verb_phrase_node = null, clause_node = null;
			clause_str = " " + tree_to_leaves_string(found_variables.get("clause")) + " ";
			np_str = tree_to_leaves_string(found_variables.get("np"));
			// if(np_str.startsWith("you")) {
			// 	System.out.println("YOU yes YOU!");
			// }
			boolean np_is_pronoun = check_if_np_is_pronoun(found_variables.get("np"));
			if(np_is_pronoun) {
				rules += "R_np_is_prp ";
			}
			is_verb_str = tree_to_leaves_string(found_variables.get("verb"));
			adverb1_str = tree_to_leaves_string(found_variables.get("adverb1"));
			adverb2_str = tree_to_leaves_string(found_variables.get("adverb2"));
			if(!adverb1_str.isEmpty()) {
				rules += "R_adv1 ";
			}
			if(!adverb2_str.isEmpty()) {
				rules += "R_adv2 ";
			}
			boolean replace_np_with_pronoun = true;

			//NOTE: assuming that adverb1 and adverb2 are mutually exclusive. Haven't found a single case where both present
			//TODO:Adverb is not of isverb but rather of VP or final verb
			//Clause children:3
			// 4310	final verb:been there to be seen
			// 4311	Clause: had already  
			// Full Question:what had already been there to be seen ?
			// Full Question Tree:(ROOT (SBARQ (WHNP (WP what)) (SQ (VBD had) (ADVP (RB already)) (VP (VBN been) (S (NP (EX there)) (VP (TO to) (VP (VB be) (VP (VBN seen))))))) (. ?)))
			// 4123	Answer1:already had been there to be seen  {the term middle east}	4314	Answer1:had already already had been there to be seen  {the term middle east}
			// 4124	Answer2: {the term middle east} had already been there to be seen	4315	Answer2: {the term middle east} already had had already been there to be seen
			// 4125	the term middle east	4316	the term middle east
			is_verb_str = adverb1_str + " " + is_verb_str;
			is_verb_str = is_verb_str.trim();
			clause_str = clause_str.replaceFirst(Pattern.quote(" " + is_verb_str + " "), "");

			// System.out.println("Clause STR:" + clause_str);
			// System.out.println("IS Verb STR:" + is_verb_str);
			clause_node = found_variables.get("clause");
			// System.out.println("Clause children:" + clause_node.children().length);
			// if(found_variables.get("final_verb") != null){
			// 	System.out.println("final_verb value:" + found_variables.get("final_verb").value());
			// 	System.exit(0);
			// }
			if(found_variables.get("final_verb") != null && found_variables.get("clause").children().length==3) {
				rules += "R_3SQ_children ";
				// System.out.println("IDHAR MILA:" + found_variables.get("final_verb"));
				if(found_variables.get("final_verb").value().equals("NP")) {
					System.out.println("Ye hai apun ka interest!");
					System.out.println(np_str);
					System.out.println(clause_str);
					System.out.println(final_verb_phrase_str);
					np_str = "";
					found_variables.put("np", null);
					rules += "R_final_NP ";
				} else {
					rules += "R_final_VP ";
				}
				final_verb_phrase_str = tree_to_leaves_string(found_variables.get("final_verb"));
				// adverb2 is already inside final_verb
				// System.out.println("final verb:" + final_verb_phrase_str);
				// Remove the final verb from clause if present
				clause_str = clause_str.replace(final_verb_phrase_str, "");
				// If there is a final verb phrase then we need to modify the clause node and update the final_verb_phrase_node

				final_verb_phrase_node = clause_node.lastChild();
				System.out.println("Clause node last child:" + final_verb_phrase_node);
				final_verb_phrase_node = found_variables.get("final_verb");
				System.out.println("Clause node last child:" + final_verb_phrase_node);
				clause_node = clause_node.children()[1];
			} else if(found_variables.get("final_verb") != null && found_variables.get("final_verb").value().equals("NP") && found_variables.get("clause").children().length==2) {
				// System.out.println("SOCHO kabhi aisa hoo toh kya hoo!!");
				// System.out.println(wh_is_question);
				// System.out.println(found_variables.get("clause"));
				// System.out.println(found_variables.get("final_verb"));
				// System.out.println(clause_str);
				// System.out.println(final_verb_phrase_str);
				// System.exit(0);
				// In this case do nothing... as we don't want final verb to be initialized
				found_variables.put("final_verb", null);
				rules += "R_2SQ_children R_no_final_verb ";
			} else if(found_variables.get("final_verb") != null) {
				rules += "R_4_or_more_SQ_children R_final_verb_exists R_possible_ERROR ";
				System.out.println("POSSIBLE ERROR:" + clause_str);
				replace_np_with_pronoun = false;
				final_verb_phrase_node = found_variables.get("final_verb");
				final_verb_phrase_str = tree_to_leaves_string(found_variables.get("final_verb"));
				clause_str = clause_str.replace(final_verb_phrase_str, "").trim();
			} else if(found_variables.get("final_verb") == null) {
				rules += "R_no_final_verb ";
			}
			// System.out.println("Clause:" + clause_str);

			// Get the PP from clause_node and final_verb_phrase_node
			ArrayList<String> prep_phrases_clause_node, prep_phrases_final_vp_node, prep_phrases_in_np_in_wh;
			prep_phrases_clause_node = get_string_from_tree_list(get_nested_pp_list(clause_node));
			if(prep_phrases_clause_node.size() > 0) {
				rules += "R_pp_in_clause ";
			}
			prep_phrases_final_vp_node = get_string_from_tree_list(get_nested_pp_list(final_verb_phrase_node));
			if(prep_phrases_final_vp_node.size() > 0) {
				rules += "R_pp_in_final_vp ";
			}
			// Add empty string in both the pp list. Used so that the code is more compact and the answers can be generated in 2 nested loops
			prep_phrases_clause_node.add("");
			prep_phrases_final_vp_node.add("");

			// Get prepositions before the wh in the wh-phrase
			found_variables = search_pattern("SBARQ=question_head < (/WH.?/=wh_phrase << " + wh_word + " ?< /NP|NN/=np_in_wh_phrase)", wh_is_question, new ArrayList<String>(Arrays.asList("question_head", "wh_phrase", "np_in_wh_phrase")));
			String preposition = "";
			String wh_phrase = tree_to_leaves_string(found_variables.get("wh_phrase"));
			String np_in_wh_phrase = tree_to_leaves_string(found_variables.get("np_in_wh_phrase")).replace(wh_word, "").trim();
			// if(np_in_wh_phrase.contains(wh_word)) {
			// 	System.out.println("WTF:" + np_in_wh_phrase);
			// 	System.out.println(found_variables.get("np_in_wh_phrase"));
			// 	System.out.println(found_variables.get("np_in_wh_phrase").lastChild());
			// }
			prep_phrases_in_np_in_wh = get_string_from_tree_list(get_nested_pp_list(found_variables.get("np_in_wh_phrase")));
			if(prep_phrases_in_np_in_wh.size() > 0) {
				rules += "R_pp_in_np_in_wh ";
			}
			if(!np_in_wh_phrase.isEmpty()) {
				// System.out.println("YEAH found NP in WH:" + np_in_wh_phrase);
				rules += "R_np_in_wh ";
				prep_phrases_in_np_in_wh.add(0, np_in_wh_phrase);
			}
			prep_phrases_in_np_in_wh.add("");
			preposition = wh_phrase.substring(0, wh_phrase.indexOf(wh_word)).trim();
			if(!preposition.isEmpty()) {
				rules += "R_prep_in_wh ";
			}

			// System.out.print(preposition);
			if(preposition.isEmpty()) {
				// check for the preposition in vp
				preposition = get_prep_from_vp(final_verb_phrase_node);
				if(!preposition.isEmpty()) {
					rules += "R_prep_in_vp ";
					System.out.println("NEW prep wtf:" + preposition);
				}
				// need to remove this prep from final_verb and its pps
				System.out.println("Final verb:" + final_verb_phrase_str);
				final_verb_phrase_str = " " + final_verb_phrase_str + " ";
				final_verb_phrase_str = final_verb_phrase_str.replaceFirst(Pattern.quote(" " + preposition + " "), " ").trim();
				System.out.println("Final verb:" + final_verb_phrase_str);
				for(int i = 0; i < prep_phrases_final_vp_node.size(); i++) {
					String pp = " " + prep_phrases_final_vp_node.get(i) + " ";
					prep_phrases_final_vp_node.set(i, pp.replaceFirst(Pattern.quote(" " + preposition + " "), " ").trim());
				}
				if(preposition.isEmpty()) {
					rules += "R_missing_prep ";
					// preposition = (wh_word.equals("why")) ? "because {missing_IN}": "{missing_IN}";
					preposition = "{missing_IN}";
				} else {
					// System.out.println("BHAIS KI TAANG.. PREP:" + preposition);
				}
			}
			System.out.print(preposition);

			// Get the subordinate clause if present
			found_variables = search_pattern("SBARQ < (SBAR=subordinate)", wh_is_question, new ArrayList<String>(Arrays.asList("subordinate")));
			String subordinate_clause = "";
			subordinate_clause = tree_to_leaves_string(found_variables.get("subordinate"));
			if(!subordinate_clause.isEmpty()) {
				rules += "R_subordinate_clause ";
				System.out.println("Ding ding ding Subordiante Clause:" + subordinate_clause);
			}

			// if(!np_str.isEmpty()) {
			// 	System.out.println("NP STR:" + np_str + " :: " + clause_str.contains(np_str));
			// 	System.out.println("Clause STR:" + clause_str);
			// }

			for(String pp_in_np_in_wh : prep_phrases_in_np_in_wh) {
				String answer_with_brackets = "{" + answer + "}" + (!np_in_wh_phrase.isEmpty()?(" " + np_in_wh_phrase.replace(pp_in_np_in_wh, "").trim()):"");
				String current_rules = rules;
				if(!np_in_wh_phrase.isEmpty() && np_in_wh_phrase.replace(pp_in_np_in_wh, "").trim().isEmpty()) {
					// np_in_wh_phrase is not added to the answer phrase
				} else if(!np_in_wh_phrase.isEmpty()){
					// np_in_wh_phrase is added to the answer phrase
					current_rules += "R_np_in_wh_added ";
				}
				for(String pp_vp : prep_phrases_final_vp_node) {
					String pre_final_rules = current_rules;
					if(pp_vp.isEmpty()) {
						pre_final_rules += "R_pp_in_vp_not_removed ";
					} else {
						// removing pp in vp so add the rule
						pre_final_rules += "R_pp_in_vp_removed ";
					}
					for(String pp_c : prep_phrases_clause_node) {
						String final_rules = pre_final_rules;
						if(pp_c.isEmpty()) {
							final_rules += "R_pp_in_clause_not_removed ";
						} else {
							// removing pp in vp so add the rule
							final_rules += "R_pp_in_clause_removed ";
						}
						if(!clause_str.isEmpty() && !clause_str.replace(pp_c, "").trim().isEmpty()) {
							// Generate the response only when the clause string after removing the PP is not empty
							final_rules += "R_clause_not_empty ";
							gen_responses_and_rules.add(new Pair(clause_str.replace(pp_c, "").trim() + " " + is_verb_str.trim().replace("\'s", "is") + " " + final_verb_phrase_str.replace(pp_vp, "").trim() + " " + preposition.trim() + " " + answer_with_brackets + "", final_rules + "R_clause_is_verb_final_verb_answer"));
							if(final_verb_phrase_node != null && final_verb_phrase_node.value().equals("NP")) {
								gen_responses_and_rules.add(new Pair(clause_str.replace(pp_c, "").trim() + " " + final_verb_phrase_str.replace(pp_vp, "").trim() + " " + is_verb_str.trim().replace("\'s", "is") + " " + preposition.trim() + " " + answer_with_brackets + "", final_rules + "R_clause_final_verb_is_verb_answer"));
							}
							if(final_verb_phrase_str.isEmpty() && !np_str.isEmpty()) {
								// Try adding the pp after the NP in clause str
								System.out.println("is after NP in clause STR:" + np_str);
								// gen_responses_and_rules.add(clause_str.replace(np_str, np_str + " " + is_verb_str.trim().replace("\'s", "is")).replace(pp_c, "").trim() + " " + final_verb_phrase_str.replace(pp_vp, "").trim() + " " + preposition.trim() + " " + answer_with_brackets + "", final_rules + "R_clause_replace_np_with_np_is_verb_answer");
								gen_responses_and_rules.add(new Pair(clause_str.replace(np_str, np_str + " " + is_verb_str.trim().replace("\'s", "is")).replace(pp_c, "").trim() + " " + preposition.trim() + " " + answer_with_brackets + "", final_rules + "R_is_verb_just_after_np R_clause_and_answer"));
							}
							if(wh_word.equals("why")) {
								gen_responses_and_rules.add(new Pair(preposition.trim() + " " + answer_with_brackets + " " + clause_str.replace(pp_c, "").trim() + " " + is_verb_str.trim().replace("\'s", "is") + " " + final_verb_phrase_str.replace(pp_vp, "").trim(), final_rules + "R_answer_clause_is_verb_final_verb"));
							} else {
								gen_responses_and_rules.add(new Pair(preposition.trim() + " " + answer_with_brackets + " " + is_verb_str.trim().replace("\'s", "is") + " " + clause_str.replace(pp_c, "").trim() + " " + final_verb_phrase_str.replace(pp_vp, "").trim(), final_rules + "R_answer_is_verb_clause_final_verb"));
							}
						} else if(clause_str.isEmpty()) {
							final_rules += "R_clause_empty ";
							System.out.println("Case 2.2");
							gen_responses_and_rules.add(new Pair(preposition.trim() + " " + answer_with_brackets + " " + is_verb_str.trim().replace("\'s", "is") + " " + final_verb_phrase_str.replace(pp_vp, "").trim(), final_rules + "R_answer_is_verb_final_verb"));
						}
					}
					if(replace_np_with_pronoun && !np_str.isEmpty() && !np_is_pronoun) {
						String final_rules = pre_final_rules + "R_np_replaced_with_pronoun ";
						for(String pronoun : all_pronouns) {
							// Generate the response only when the clause string after removing the PP is not empty
							gen_responses_and_rules.add(new Pair(clause_str.replace(np_str, pronoun).trim() + " " + is_verb_str.trim().replace("\'s", "is") + " " + final_verb_phrase_str.replace(pp_vp, "").trim() + " " + preposition.trim() + " " + answer_with_brackets, final_rules + "R_clause_is_verb_final_verb_answer"));
							if(wh_word.equals("why")) {
								gen_responses_and_rules.add(new Pair(preposition.trim() + " " + answer_with_brackets + " " + clause_str.replace(np_str, pronoun).trim() + " " + is_verb_str.trim().replace("\'s", "is") + " " + final_verb_phrase_str.replace(pp_vp, "").trim(), final_rules + "R_answer_clause_is_verb_final_verb"));
							} else {
								gen_responses_and_rules.add(new Pair(preposition.trim() + " " + answer_with_brackets + " " + is_verb_str.trim().replace("\'s", "is") + " " + clause_str.replace(np_str, pronoun).trim() + " " + final_verb_phrase_str.replace(pp_vp, "").trim(), final_rules + "R_answer_is_verb_clause_final_verb"));
							}
						}
					}
				}
			}
		}

		return gen_responses_and_rules;
	}
	// public static ArrayList<Pair> what_did_question_to_answer(Tree what_did_question, String answer) {
		// TODO: sometimes for will can and would type of questions the answer phrase should be before the verb rather than after the verb. Consider shifting this to a different function altogether

		// TODO: In one exmaple ADVP is a separate node from the VP and that is being ignored in the current system. Think about a way to fix it
		// 		 Example: (SBARQ (WHPP (IN on) (WHNP (WDT what) (NN date))) (SQ (VBD did) (NP (NP (DT the) (NN playstation)) (CC plus) (NP (NN service))) (ADVP (RB officially)) (VP (VB launch))) (. ?))

		// TODO: If the secondary verb is do then the answer should be a verb and therefore do should be replaced with answer instead.
		// NOTE: implementing this breaks some cases so better not implement this do case. Should work if the answer is verb but if it is not then we should ignore this rule.
		// TODO: How to detect if a word is a verb?

		/*If the main verb is “did”. 
		- Take the main wh-question phrase SQ. Check the tense of the main verb (VP VBD) and change the form of subordinate verb’s (inside the subordinate clause) tense based on it. 
		- Append the answer after the verb
		- append/prepend/ignore the other subordinate clause to make the whole sentence
		*/


	/*Full Question:what world leader notably visited libya in 2002 ?
	Full Question Tree:(SBARQ (WHNP (WDT what) (NN world) (NN leader)) (SQ (ADVP (RB notably)) (VP (VBD visited) (NP (NN libya)) (PP (IN in) (NP (CD 2002))))) (. ?))
	Answer1:{jiang zemin} world leader notably visited libya
	Answer2:{jiang zemin} world leader notably visited libya in 2002*/

	public static ArrayList<Pair> wh_verb_question_to_answer(Tree wh_verb_question, String answer, String wh_word) {
		/*If neither “did” or “is” is the main verb then, 
		Replace answer with WH-phrase (including the noun phrase)
		and optionally prepend/append/ignore the subordinating clause.
		*/
		// TODO: can this be changed to is <verb> by
		ArrayList<Pair> gen_responses_and_rules = new ArrayList<Pair>();
		HashMap<String, Tree> found_variables;
		String rules = "R_3 ";			// We will store the sequence code path or used rules in this string. Will be later used as a feature in the models 
		// Get the WH-phrase and the adjacent SQ
		found_variables = search_pattern("SBARQ < (/WH.?/=wh_phrase) < (SQ=adjacent_sq)", wh_verb_question, new ArrayList<String>(Arrays.asList("wh_phrase", "adjacent_sq")));
		String wh_phrase = "", adjacent_sq = "", full_question = tree_to_leaves_string(wh_verb_question);
		wh_phrase = tree_to_leaves_string(found_variables.get("wh_phrase"));
		adjacent_sq = tree_to_leaves_string(found_variables.get("adjacent_sq"));
		if(adjacent_sq.isEmpty()) {
			// There is an error in the parse tree
			// For example: (ROOT (SBARQ (WHADVP (WRB where)) (SBARQ (MD can) (SQ (VBD advanced) (NP (NNS settings) (NNS options)) (VP (VB be) (VP (VBN found))))) (. ?)))
			// Simply ignore such cases
			return gen_responses_and_rules;
		}
		System.out.println("###wh phrase:" + wh_phrase);
		System.out.println("###Adjacent SQ:" + adjacent_sq);

		// Get the PPs from the clause if present
		// PPs can exist inside a WH-phrase
		//TODO: in the data I've only seen PP in the as a subordinate clause. Could be something else in read data. Verify
		ArrayList<String> prep_phrases_clause_node, prep_phrases_wh_node, prep_phrases_in_np_in_wh;
		System.out.println("SQ children:" + found_variables.get("adjacent_sq").children().length);
		prep_phrases_clause_node = get_string_from_tree_list(get_nested_pp_list(found_variables.get("adjacent_sq")));
		prep_phrases_wh_node = get_string_from_tree_list(get_nested_pp_list(found_variables.get("wh_phrase")));
		if(!prep_phrases_clause_node.isEmpty()) {
			rules += "R_pp_in_clause ";
			System.out.println("PPs Clause:" + prep_phrases_clause_node);
		}
		if(!prep_phrases_wh_node.isEmpty()) {
			rules += "R_pp_wh ";
			System.out.println("PPs Wh node:" + prep_phrases_wh_node);
		}
		// Add extra empty element to the PPs list so that we can do everything in 2 nested loops
		prep_phrases_clause_node.add("");
		prep_phrases_wh_node.add("");

		// Get prepositions before the wh in the wh-phrase
		found_variables = search_pattern("SBARQ=question_head < (/WH.?/=wh_phrase << " + wh_word + " ?< /NP|NN/=np_in_wh_phrase)", wh_verb_question, new ArrayList<String>(Arrays.asList("question_head", "wh_phrase", "np_in_wh_phrase")));
		String preposition = "";
		String wh_phrase2 = tree_to_leaves_string(found_variables.get("wh_phrase"));
		preposition = wh_phrase2.substring(0, wh_phrase2.indexOf(wh_word)).trim();
		String np_in_wh_phrase = tree_to_leaves_string(found_variables.get("np_in_wh_phrase")).replace(wh_word, "").trim();
		prep_phrases_in_np_in_wh = get_string_from_tree_list(get_nested_pp_list(found_variables.get("np_in_wh_phrase")));
		if(prep_phrases_in_np_in_wh.size() > 0) {
			rules += "R_pp_in_np_in_wh ";
		}
		if(!np_in_wh_phrase.isEmpty()) {
			System.out.println("YEAH found NP in WH:" + np_in_wh_phrase);
			prep_phrases_in_np_in_wh.add(0, np_in_wh_phrase);
		}
		prep_phrases_in_np_in_wh.add("");
		if(preposition.isEmpty()) {
			// TODO: think about if we want to insert a preposition here?
			preposition = "{missing_IN}";
		} else {
			// System.out.println("PREP FOUND");
		}

		// if(!subordinate_clause.isEmpty()) {
		// 	gen_responses_and_rules.add(subordinate_clause + " , {" + answer + "} " + adjacent_sq);
		// 	gen_responses_and_rules.add("{" + answer + "} " + adjacent_sq + " , " + subordinate_clause);
		// }
		for(String pp_in_np_in_wh : prep_phrases_in_np_in_wh) {
			String answer_with_brackets = "{" + answer + "}" + (!np_in_wh_phrase.isEmpty()?(" " + np_in_wh_phrase.replace(pp_in_np_in_wh, "").trim()):"");
			String current_rules = rules;
			if(!np_in_wh_phrase.isEmpty() && np_in_wh_phrase.replace(pp_in_np_in_wh, "").trim().isEmpty()) {
				// np_in_wh_phrase is not added to the answer phrase
			} else if(!np_in_wh_phrase.isEmpty()){
				// np_in_wh_phrase is added to the answer phrase
				current_rules += "R_np_in_wh_added ";
			}
			for(String pp_c : prep_phrases_clause_node) {
				String final_rules = current_rules;
				if(pp_c.isEmpty()) {
					final_rules += "R_pp_in_clause_not_removed ";
				} else {
					// removing pp in vp so add the rule
					final_rules += "R_pp_in_clause_removed ";
				}
				gen_responses_and_rules.add(new Pair(preposition + " " + answer_with_brackets + " " + adjacent_sq.replaceFirst(Pattern.quote(pp_c), ""), final_rules + "R_answer_clause_pp"));
				if(!pp_c.isEmpty()) {
					gen_responses_and_rules.add(new Pair(pp_c + ", " + preposition + " " + answer_with_brackets + " " + adjacent_sq.replaceFirst(Pattern.quote(pp_c), ""), final_rules + "R_pp_answer_clause"));
				}
			}
		}
		// if(!subordinate_clause2.isEmpty()) {
		// 	gen_responses_and_rules.add(subordinate_clause2 + " , {" + answer + "} " + adjacent_sq);
		// 	gen_responses_and_rules.add("{" + answer + "} " + adjacent_sq + " , " + subordinate_clause2);
		// }
		// gen_responses_and_rules.add("{" + answer + "} " + adjacent_sq);

		return gen_responses_and_rules;
	}

	public static ArrayList<Pair> wh_question_to_answer(Tree wh_question, String answer, String wh_word) {
		ArrayList<Pair> gen_responses_and_rules = new ArrayList<Pair>();
		HashMap<String, Tree> found_variables;
		boolean match_found = false;
		// If SQ has only one child which is VP then excise it

		// RULE 1
		// If the wh question is a (did/do/does/do not/did not) question:
		TregexPattern did_question_pattern = TregexPattern.compile("SBARQ < (SQ=main_clause (< (/VB.?/ <, (do|does|did)) | < (MD < (will|would|can|must|may|should|could))) )");
		TregexMatcher did_matcher = did_question_pattern.matcher(wh_question);
		if(did_matcher.find()) {
			// Send this to did handler
			gen_responses_and_rules.addAll(wh_did_question_to_answer(wh_question, answer, wh_word));
			match_found = true;
		}
		// RULE 2
		// If the main verb is “is|was”.
		TregexPattern is_question_pattern = TregexPattern.compile("SBARQ=question_head < (SQ=main_clause (< (/VB.?/ < (is|\'s|are|has|have|had|was|were))) )");
		TregexMatcher is_matcher = is_question_pattern.matcher(wh_question);
		if(is_matcher.find() && !match_found) {
			// Send this to is handler
			gen_responses_and_rules.addAll(wh_is_question_to_answer(wh_question, answer, wh_word));
			match_found = true;
		}
		// RULE 3
		// If the main verb is neither "is|was" nor "did/do/does"
		found_variables = search_pattern("SBARQ=question_head < (SQ=main_clause < (/VB.?/ !< (is|\'s|are|has|have|had|was|were|did|do|does|will|would|can|must|may|should|could)))", wh_question, new ArrayList<String>(Arrays.asList("question_head", "main_clause")));
		if(!found_variables.isEmpty() && !match_found) {
			// Send this to verb handler
			System.out.println(wh_word + " Question Rule 3.1");
			gen_responses_and_rules.addAll(wh_verb_question_to_answer(wh_question, answer, wh_word));
			match_found = true;
		}
		// found_variables = search_pattern("SBARQ=question_head < (SQ=main_clause < (VP < (/VB.?/ !< (is|\'s|are|has|have|had|was|were|did|do|does|will|would|can))))", wh_question, new ArrayList<String>(Arrays.asList("question_head", "main_clause")));
		// if(!found_variables.isEmpty() && gen_responses_and_rules.isEmpty()) {
		// 	// Even this case should be handled by Rule 3
		// 	System.out.println("Wh Question Rule 3.2");
		// 	gen_responses_and_rules.addAll(wh_verb_question_to_answer(wh_question, answer));
		// }

		// RULE 4
		// If anything else that has wh in it. Just replace the wh with the answer
		if(gen_responses_and_rules.isEmpty()) {
			String full_question = " " + tree_to_leaves_string(wh_question).toLowerCase() + " ";
			String wh_phrase = " " + wh_word + " ";
			String rules = "R_4 ";
			// They can have PPs under the SQ. So find the PPs and optionally remove them to create shorter replies
			System.out.println(wh_word + " Question Rule 4");
			found_variables = search_pattern("SBARQ ?< /WH.?/=wh_phrase < (SQ=main_clause ?<, NP=first_np)", wh_question, new ArrayList<String>(Arrays.asList("main_clause", "wh_phrase", "first_np")));
			if(found_variables.get("wh_phrase") != null) {
				System.out.println(wh_word + " phrase:" + found_variables.get("wh_phrase") + found_variables.get("wh_phrase").firstChild());
				// Remove the preposition from the wh phrase if present because that can be an important preposition
				// Example: (WHPP (IN through) (WHNP (WDT wh) (NN period)))
				String prep_within_phrase = "";
				if(found_variables.get("wh_phrase").firstChild().value().equals("IN") || found_variables.get("wh_phrase").firstChild().value().equals("TO")) {
					rules += "R_prep_in_wh ";
					prep_within_phrase = tree_to_leaves_string(found_variables.get("wh_phrase").firstChild());
					// System.out.println("PP FIRST CHILD:" + prep_within_phrase);
				}
				wh_phrase = " " + tree_to_leaves_string(found_variables.get("wh_phrase")).replace(prep_within_phrase, "").trim() + " ";
			}
			// If SQ has first child as NP. In all the cases that NP should actually be a part of WHNP
			// Examples: (SBARQ (WHNP (WP wh)) (SQ (NP (JJ precious) (NN material)) (VP (MD may) (VP (VB remain) (ADJP (JJ uncut))))) (. ?))
			//			 (SBARQ (WHNP (WP wh)) (SQ (NP (JJ big) (NN cat)) (VP (VBZ has) (NP (DT a) (NN tendency) (S (VP (TO to) (VP (VB attack) (NP (NNS dogs)))))))) (. ?))
			if(found_variables.get("first_np") != null) {
				rules += "R_first_NP_fix ";
				System.out.println("FIRST NP:" + wh_phrase + tree_to_leaves_string(found_variables.get("first_np")));
				wh_phrase += tree_to_leaves_string(found_variables.get("first_np"));
				wh_phrase = wh_phrase.trim();
			}
			ArrayList<String> prep_phrases = get_string_from_tree_list(get_nested_pp_list(found_variables.get("main_clause")));
			if(prep_phrases.size() > 0) {
				rules += "R_pp_in_clause ";
			}
			// Add an emtpy string to this list so that we can generate all answers in one loop
			prep_phrases.add("");
			for(String pp : prep_phrases) {
				String final_rules = rules;
				if(pp.isEmpty()) {
					final_rules += "R_pp_in_clause_not_removed";
				} else {
					final_rules += "R_pp_in_clause_removed";
				}
				if(full_question.contains(wh_phrase)) {
					gen_responses_and_rules.add(new Pair(full_question.replace("?", "").replace(wh_phrase, " {" + answer + "} ").replace(pp, "").trim(), final_rules));
				} else {
					System.out.println("ERROR: " + wh_word + " question without the string " + wh_word + "-phrase");
				}
			}
		}

		gen_responses_and_rules = remove_duplicate_responses(gen_responses_and_rules);
		// print_all_responses(gen_responses_and_rules);
		return gen_responses_and_rules;
	}

	// Sometimes the part of the sentence after the main verb is more suitable before the answer
	// Examples: (ROOT (SBARQ (WHNP (WP who)) (SQ (VBD did) (NP (JJ anglophone) (NNS colonies)) (ADVP (RB democratically)) (VP (VB glorify) (NP (NN hunting)) (PP (IN for)))) (. ?)))
	//			 (ROOT (SBARQ (WHNP (WP who)) (SQ (VBD did) (NP (DT the) (JJ central) (NN junta)) (VP (VB fall) (PP (TO to)))) (. ?)))

	
	// (ROOT (SBARQ (WHADVP (WRB when)) (SQ (VBD did) (NP (NNP beyoncé)) (VP (VB give) (NP (NN birth)) (PP (TO to) (NP (PRP$ her) (NN daughter))))) (. ?)))



	// DONE: consider keeping the NP in the WH phrase with the answer. Sometimes its needed
	// Example: Full Question Tree:(ROOT (SBARQ (WHNP (WDT which) (NN name)) (SQ (VBD did) (NP (NN russia)) (VP (VB take) (PP (IN after) (NP (NP (DT the) (NN fall)) (PP (IN of) (NP (DT the) (JJ soviet) (NN union))))))) (. ?)))

	// TODO: fix the missing prep in this case
	// Full Question:which age did the invention of the papermaking process contribute towards ?
	// Full Question Tree: (ROOT (SBARQ (WHNP (WDT which) (NN age)) (SQ (VBD did) (NP (NP (DT the) (NN invention)) (PP (IN of) (NP (DT the) (JJ papermaking) (NN process)))) (VP (VB contribute) (PP (IN towards)))) (. ?)))

	//TODO: Find prepositions which are in the final verb phrase after the final verb
	//Example: (ROOT (SBARQ (WHNP (WDT which) (JJ other) (NN system)) (SQ (VBD were) (NP (DT these) (NNS changes)) (VP (VBN applied) (PP (TO to)))) (. ?)))
	//Another Example: (ROOT (SBARQ (WHADVP (WRB where)) (SQ (VBD did) (NP (NP (JJ princess) (NN victoria)) (CC and) (NP (PRP$ her) (NN husband))) (VP (VB leave) (PP (IN for) (PP (IN after) (NP (PRP$ their) (NN marriage)))))) (. ?)))



	// TODO: think of removing the is_verb from the answer phrase for this sentence

	//TODO: For why question we might have to change {missing_IN} with "because {missing_IN}"
	// NOTE: Haven't found a single question for rule 3. Might be incorrect

	//TODO: why was PP not found in this case?
	// Main verb initially:(VB get)
	// Verb Phrase:get in the first four days after it was posted
	// Main verb: get
	// Changed Verb: got
	// Full Question:how many views did grimm 's last email get in the first four days after it was posted ?
	// Full Question Tree:(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) (NNS views)) (SQ (VBD did) (NP (NP (NN grimm) (POS 's)) (JJ last) (NN email)) (VP (VB get) (PP (IN in) (NP (DT the) (JJ first) (CD four) (NNS days))) (SBAR (IN after) (S (NP (PRP it)) (VP (VBD was) (VP (VBN posted))))))) (. ?)))
	// Answer1:{missing_IN} {more than 100,000} , grimm 's last email   got in the first four days  
	// Answer2:{missing_IN} {more than 100,000} , grimm 's last email   got in the first four days after it was posted 
	// Answer3:grimm 's last email   got in the first four days   {missing_IN} {more than 100,000} 
	// Answer4:grimm 's last email   got in the first four days after it was posted  {missing_IN} {more than 100,000} 
	// more than 100,000


	// TODO: fix this case
	// Main verb initially:(VB cover)
	// Verb Phrase:cover
	// Main verb: cover
	// Changed Verb: covers
	// Full Question:how many counties does oklahoma cities networks cover ?
	// Full Question Tree:(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) (NNS counties)) (SQ (VBZ does) (NP (JJ oklahoma) (NNS cities)) (NP (NNS networks)) (VP (VB cover))) (. ?)))
	// Answer1:{missing_IN} {34} , networks   covers 
	// Answer2:networks   covers  {missing_IN} {34} 
	// 34

	// TODO: think of removing the is_verb from the answer phrase for this sentence
	// TODO: haven't analyses RULE 3 fully

	public static ArrayList<Pair> insert_missing_prepositions(ArrayList<Pair> gen_responses_and_rules, MaxentTagger tagger) {
		ArrayList<Pair> new_gen_responses_and_rules = new ArrayList<Pair>();
		String[] temp_split;
		for(Pair response_and_rule: gen_responses_and_rules) {
			String response = response_and_rule.response;
			String rule = response_and_rule.rule;
			// find the position of missing_IN if present
			if(response.contains("{missing_IN}")) {
				response = response.trim();
				String response_without_missing_prep = response.replace("{missing_IN}", "").trim();
				// We will first remove the "{missing_IN}" from the response and then POS tag it to see if it already as a preposition (maybe inside the answer phrase or we somehow missed it)
				// If it has then no need to add the preposition
				String response_without_missing_prep_and_answer_brackets = response_without_missing_prep.replace("{", "").replace("}", "");
				// POS tag 
				String tagged_response_without_missing_prep_and_answer_brackets = tagger.tagString(response_without_missing_prep_and_answer_brackets);
				// Find the position of answer start
				temp_split = response_without_missing_prep.split("\\s+");
				int answer_start_pos = -1, answer_end_pos = -1;
				for(int i = 0; i < temp_split.length; i++) {
					String current_word = temp_split[i];
					if(current_word.startsWith("{")) {
						answer_start_pos = i;
					}
					if(current_word.endsWith("}")) {
						answer_end_pos = i;
					}
				}
				if(answer_end_pos < answer_start_pos && answer_start_pos == -1) {
					// Some serious error
					System.out.println("SERIOUS ERROR: answer not found in the response!! Exitting");
					System.out.println(response);
					System.exit(1);
				}
				// Check if the first answer word or the word before it is a Preposition i.e. either tagged with IN or TO
				temp_split = tagged_response_without_missing_prep_and_answer_brackets.split("\\s+");
				// System.out.println("Pos tagged Response:" + tagged_response_without_missing_prep_and_answer_brackets);
				// System.out.println("Answer Start Pos:" + answer_start_pos);
				boolean prep_present = false;
				boolean shift_answer_start = false;
				if(answer_start_pos == 0) {
					String check_pos_tag = temp_split[0].split("_")[1].trim();
					String check_word = temp_split[0].split("_")[0].trim();
					// IN is for preposition and subordinate conjunctions and we should disregard "because" as it is not a preposition
					// TODO: Allowing because in the prep as we may find cases with because inserted before a because in the answer phrase
					// prep_present = (check_pos_tag.equals("IN") && !check_word.equals("because")) || check_pos_tag.equals("TO");
					prep_present = (check_pos_tag.equals("IN")) || check_pos_tag.equals("TO");
					if(prep_present && answer_start_pos != answer_end_pos) {
						shift_answer_start = true;
					}
				} else if(answer_start_pos > 0) {
					System.out.println(answer_start_pos + " : " + temp_split.length);
					System.out.println(tagged_response_without_missing_prep_and_answer_brackets);
					System.out.println(response);
					String check_pos_tag_1 = temp_split[answer_start_pos].split("_")[1].trim();
					String check_word_1 = temp_split[answer_start_pos].split("_")[0].trim();
					String check_pos_tag_2 = temp_split[answer_start_pos - 1].split("_")[1].trim();
					String check_word_2 = temp_split[answer_start_pos - 1].split("_")[0].trim();
					// prep_present = (check_pos_tag_1.equals("IN") && !check_word_1.equals("because")) || check_pos_tag_1.equals("TO") || (check_pos_tag_2.equals("IN") && !check_word_2.equals("because")) || check_pos_tag_2.equals("TO");
					prep_present = (check_pos_tag_1.equals("IN")) || check_pos_tag_1.equals("TO") || (check_pos_tag_2.equals("IN")) || check_pos_tag_2.equals("TO");
					if(((check_pos_tag_1.equals("IN")) || check_pos_tag_1.equals("TO")) && answer_start_pos != answer_end_pos) {
						shift_answer_start = true;
					}
				} else {
					// Answer start pos is -1. This means the generated response is without answer which is unacceptable.
					// Report error and terminate
					System.out.println("Error: The generated response is wihout the answer phrase:" + response);
					System.exit(0);
				}
				
				// System.out.println(tagged_response_without_missing_prep_and_answer_brackets + " :: " + temp_split.length);
				if(prep_present) {
					String final_rule = rule + " R_prep_already_present_in_ans";
					// Simply add the response after replacing "{missing_IN}" as preposition is already present
					String new_response = response_without_missing_prep;
					temp_split = response_without_missing_prep.split("\\s+");
					if(shift_answer_start) {
						final_rule += " R_shifting_ans_start_bracket";
						// System.out.println("Answer start shifted!");
						new_response = "";
						for(int i = 0; i < temp_split.length; i++) {
							if(i == answer_start_pos && temp_split.length > (answer_start_pos+1)) {
								// Need to shift answer start position to next word.. if present
								new_response += temp_split[i].replace("{", "") + " ";
							} else if(i == (answer_start_pos+1)) {
								new_response += "{" + temp_split[i] + " ";
							} else {
								new_response += temp_split[i] + " ";
							}
						}
						// System.out.println(response);
						// System.out.println(answer_start_pos + " : " + answer_end_pos);
						// System.out.println(response_without_missing_prep);
						// System.out.println(new_response);
					}
					new_gen_responses_and_rules.add(new Pair(new_response.replace("because because", "because").replaceAll("\\s+", " ").trim(), final_rule));
				} else {
					// replace the {missing_IN} with all the possible prepositions from the list
					for(String prep : all_prepositions) {
						String final_rule = rule;
						if(prep.isEmpty()) {
							final_rule += " R_no_prep_added";
						} else {
							final_rule += " R_custom_prep_added";
						}
						new_gen_responses_and_rules.add(new Pair(response.replace("{missing_IN}", prep).replace("because because", "because").replaceAll("\\s+", " "), final_rule));
					}
				}
			} else {
				// add the response as it is
				new_gen_responses_and_rules.add(response_and_rule);
			}
		}
		return new_gen_responses_and_rules;
	}

	
	// NOTE: call this function after interesting the prepositions
	public static ArrayList<Pair> insert_missing_determiner(ArrayList<Pair> gen_responses_and_rules, MaxentTagger tagger) {
		// We know the prep's position. We need to check if the next word is a determiner or not
		ArrayList<Pair> new_gen_responses_and_rules = new ArrayList<Pair>();
		String[] temp_split;
		for(Pair response_and_rule: gen_responses_and_rules) {
			String response = response_and_rule.response;
			String rule = response_and_rule.rule;
			response = response.toLowerCase().trim();
			// find the answer pos after splitting the response
			temp_split = response.trim().split("\\s+");
			int answer_start_pos = -1;
			int answer_end_pos = -1;
			for(int i = 0; i < temp_split.length; i++) {
				String current_word = temp_split[i];
				if(current_word.startsWith("{")) {
					answer_start_pos = i;
				}
				if(current_word.endsWith("}")) {
					answer_end_pos = i;
				}
			}
			String response_without_answer_brackets = response.trim().replace("{", "").replace("}", "");
			// POS tag 
			String tagged_response_answer_brackets = tagger.tagString(response_without_answer_brackets);
			
			// System.out.println(tagged_response_answer_brackets);
			// System.out.println(answer_start_pos + " : " + answer_end_pos);
			// Check if answer pos or answer pos + 1 is a determiner
			boolean det_present = false, det2_present = false;
			temp_split = tagged_response_answer_brackets.split("\\s+");
			for(int i = 0; i < temp_split.length; i++) {
				if((i == answer_start_pos || i == (answer_start_pos+1)) && (i <= answer_end_pos) && temp_split[i].split("_")[1].equals("DT")) {
					det_present = true;
					det2_present = (i == (answer_start_pos+1));
				}
			}
			if(det_present) {
				rule += " R_det_already_present";
				// Don't do anything as determiner already present
				// if(det2_present)
				// System.out.println("DET ALREADY PRESENT:" + response);
				// System.out.println("Answer start pos:" + answer_start_pos);
			} else {
				// Add det before the answer start
				temp_split = response.trim().toLowerCase().split("\\s+");
				boolean use_a = true;
				ArrayList<String> bracket_vowels = new ArrayList<String>(Arrays.asList("{a", "{e", "{i", "{o", "{u"));
				for(String bracket_vowel : bracket_vowels) {
					// System.out.println(temp_split.length + " : " + answer_start_pos + " : " + answer_end_pos);
					// System.out.println(response);
					if(temp_split[answer_start_pos].startsWith(bracket_vowel)) {
						use_a = false;
					}
				}
				for(String DET : all_dets) {
					if((use_a && DET.equals("an")) || (!use_a && DET.equals("a"))) {
						continue;
					}
					String new_response = "";
					for(int i = 0; i < temp_split.length; i++) {
						if (i == answer_start_pos) {
							new_response += DET + " ";
						}
						new_response += temp_split[i] + " ";
					}
					new_gen_responses_and_rules.add(new Pair(new_response.replaceAll("\\s+", " ").trim(), rule + " R_custom_det_added"));
				}
			}
			// Keep the original response without det as well
			new_gen_responses_and_rules.add(new Pair(response.replaceAll("\\s+", " ").trim(), rule));
		}
		return new_gen_responses_and_rules;
	}

	public static ArrayList<Pair> fix_special_characters(ArrayList<Pair> gen_responses_and_rules) {
		for(int i = 0; i < gen_responses_and_rules.size(); i++) {
			Pair response_and_rule = gen_responses_and_rules.get(i);
			String response = response_and_rule.response;
			String rule = response_and_rule.rule;
			response = response.replace("-lrb-", "(").replace("-rrb-", ")");
			gen_responses_and_rules.set(i, new Pair(response, rule));
		}
		// TODO: want to add responses with no parenthesis.. Except within the answer phrase
		Pattern pattern = Pattern.compile("\\(.+\\)");
		Matcher matcher; 
		ArrayList<Pair> new_gen_responses_and_rules = new ArrayList<Pair>(gen_responses_and_rules);
		for(Pair response_and_rule : gen_responses_and_rules) {
			String response = response_and_rule.response;
			String rule = response_and_rule.rule;
			// Add the response to the new list as it is
			// System.out.println(response);
			// System.out.println(response.indexOf("{") + " : " + response.indexOf("}") + " : " + response.length());
			String answer_phrase = response.substring(response.indexOf("{"), response.indexOf("}")+1);
			String response_without_answer_phrase = response.replace(answer_phrase, "{}");
			matcher = pattern.matcher(response_without_answer_phrase);
			if(matcher.find()) {
				String new_reponse = response_without_answer_phrase.substring(0, matcher.start()) + response_without_answer_phrase.substring(matcher.end());
				new_reponse = new_reponse.replaceAll("\\s+", " ").replace("{}", answer_phrase);
				// System.out.println("After removing BRACKETS:" + new_reponse);
				new_gen_responses_and_rules.add(new Pair(new_reponse, rule + " R_removed_bracketed_info"));
			}
			for(String special_case : all_special_cases) {
				if(response.contains(special_case)) {
					new_gen_responses_and_rules.add(new Pair(response.replaceFirst(Pattern.quote(special_case), "").trim().replaceAll("\\s+", " "), rule + " R_removed_special_case"));
				}
			}
		}
		return new_gen_responses_and_rules;
	}

	public static ArrayList<Pair> question_to_answer(Tree question, String answer, MaxentTagger tagger) {
		// System.out.println("\n\nquestion:" + question.toString());
		ArrayList<Pair> gen_responses_and_rules = new ArrayList<Pair>();
		HashMap<String, Tree> found_variables;

		question = remove_only_child(question);
		
		for(String wh_question : all_wh_questions) {
			found_variables = search_pattern("SBARQ=question_head < (/WH.?/=wh_phrase << " + wh_question + ") ", question, new ArrayList<String>(Arrays.asList("question_head", "wh_phrase")));
			if(!found_variables.isEmpty()){
				Tree question_head = Tree.valueOf(""), wh_phrase;
				question_head = found_variables.get("question_head");
				wh_phrase = found_variables.get("wh_phrase");

				gen_responses_and_rules.addAll(wh_question_to_answer(question_head, answer, wh_question));
				if(gen_responses_and_rules.size() > 0) {
					// already found some answers. Quit further processing.
					break;
				}
			}
		}

		// Insert missing prepositions
		gen_responses_and_rules = insert_missing_prepositions(gen_responses_and_rules, tagger);
		gen_responses_and_rules = insert_missing_determiner(gen_responses_and_rules, tagger);
		gen_responses_and_rules = fix_special_characters(gen_responses_and_rules);
		if(!gen_responses_and_rules.isEmpty()) {
			System.out.println("Full Question:" + tree_to_leaves_string(question));
			System.out.println("Full Question Tree:" + question);
			// print_all_responses(gen_responses_and_rules);
			System.out.println(answer + "\n");
		} else {
			System.out.println("System failed for question:" + question);
		}
		return gen_responses_and_rules;
	}

	public static String lowercase_tree_string(String tree_string) {
		if(tree_string == null) {
			return null;
		}
		// We need to lowercase the words within tree string because rest of the code doesn't handle it properly
		StringBuffer sb = new StringBuffer();
		String regex = "\\s[\\p{L}\\-]+\\)";
		Pattern pattern1 = Pattern.compile(regex);
		Matcher m = pattern1.matcher(tree_string);
		while(m.find()) {
			// System.out.print(m.group(0) + "::");
			m.appendReplacement(sb, m.group(0).toLowerCase());
		}
		m.appendTail(sb);

		// System.out.println("LOWERCASE TREE:" + sb.toString());
		return sb.toString();
	}

	public static boolean verify_tree_string(String tree_string, String q_str) {
		// We need to lowercase the words within tree string because rest of the code doesn't handle it properly
		StringBuffer sb = new StringBuffer();
		String regex = "\\s([\\p{L}&0-9/\\+\\;\\$\\!\\#%\\:\\`\\?\\.\\,\\'\\-]+)\\)";
		Pattern pattern1 = Pattern.compile(regex);
		Matcher m = pattern1.matcher(tree_string);
		while(m.find()) {
			// System.out.print(m.group(0) + "::");
			sb.append(m.group(1).toLowerCase());
		}

		// System.out.println("LOWERCASE TREE:" + sb.toString());
		String q_from_tree_string = sb.toString().trim().replace("`", "'").replace("''","\"").replaceFirst("\\?","");
		q_str = q_str.toLowerCase().replace(" ","").replace("''", "\"").replaceFirst("\\?","").trim();
		System.out.println("Tree string:" + tree_string);
		System.out.println(q_from_tree_string);
		System.out.println(q_str);
		return q_str.equals(q_from_tree_string);
	}
	//TODO: handle you questions
	//TODO: optionally add the NP within wh-phrase to the answer phrase. Also think how to remove "type of", "layer of", "percentage of" type of sub-phrase within the np_within_wh
	//Question: what event marks the beginning of the great depression ?
	//Answer: the stock market crash of october 29 , 1929

	public static void main(String[] args) throws IOException{
		// Initializing the NIHLexicon for SimpleNLG
		
		lexicon = new NIHDBLexicon(DB_FILENAME);
		// Setup the SimpleNLG engine
		NLGFactory nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);

		// String questions_tree_file = "squad_analysis_lexicalized_inline.txt";
		// String questions_file = "squad_analysis_inline.txt";
		// String answers_file = "squad_analysis_answer_inline.txt";
		// String questions_save_file = "rule_based_system_saved_questions.txt";
		// String generated_answer_file = "rule_based_system_generated_answers.txt";

		// SQUAD 6000 sample case in-sensitive questions
		// String questions_tree_file = "squad_train_sample/squad_train_questions_6000_sample_lexparsed_inline.txt";
		// String questions_file = "squad_train_sample/squad_train_questions_6000_sample.txt";
		// String answers_file = "squad_train_sample/squad_train_data_6000_sample_answers.txt";
		// String questions_save_file = "squad_train_sample/rule_based_system_squad_train_sample_responses_saved_questions.txt";
		// String generated_responses_file = "squad_train_sample/rule_based_system_squad_train_sample_responses_generated_answers.txt";
		// String generated_rules_file = "squad_train_sample/rule_based_system_squad_train_sample_responses_generated_answer_rules.txt";

		// // SQUAD 6000 sample case sensitive questions
		// String questions_tree_file = "squad_train_sample/squad_train_questions_6000_sample_case_sensitive_lexparsed_inline.txt";
		// String questions_file = "squad_train_sample/squad_train_questions_6000_sample_case_sensitive.txt";
		// String answers_file = "squad_train_sample/squad_train_data_6000_sample_case_sensitive_answers.txt";
		// String questions_save_file = "squad_train_sample/rule_based_system_squad_train_sample_case_sensitive_responses_saved_questions.txt";
		// String generated_responses_file = "squad_train_sample/rule_based_system_squad_train_sample_case_sensitive_responses_generated_answers.txt";
		// String generated_rules_file = "squad_train_sample/rule_based_system_squad_train_sample_case_sensitive_responses_generated_answer_rules.txt";

		// // SQUAD 6000 sample case sensitive questions debug
		// String questions_tree_file = "squad_train_sample/squad_train_questions_6000_sample_case_sensitive_lexparsed_inline.txt";
		// String questions_file = "squad_train_sample/squad_train_questions_6000_sample_case_sensitive.txt";
		// String answers_file = "squad_train_sample/squad_train_data_6000_sample_case_sensitive_answers.txt";
		// String questions_save_file = "squad_train_sample/rule_based_system_squad_train_sample_case_sensitive_responses_saved_questions.txt";
		// String generated_responses_file = "squad_train_sample/rule_based_system_squad_train_sample_case_sensitive_responses_generated_answers.txt";
		// String generated_rules_file = "squad_train_sample/rule_based_system_squad_train_sample_case_sensitive_responses_generated_answer_rules.txt";
		
		// Natural Questions simplified train set case in-sensitive questions
		// String questions_tree_file = "natural_questions/natural_questions_simplified_train_question_lexparsed_inline.txt";
		// String questions_file = "natural_questions/natural_questions_simplified_train_question.txt";
		// String answers_file = "natural_questions/natural_questions_simplified_train_short_answers.txt";
		// String questions_save_file = "natural_questions/rule_based_system_natural_questions_train_short_answers_case_insensitive_responses_saved_questions.txt";
		// String generated_responses_file = "natural_questions/rule_based_system_natural_questions_train_short_answers_case_sensitive_responses_generated_answers.txt";
		// String generated_rules_file = "natural_questions/rule_based_system_natural_questions_train_short_answers_case_sensitive_responses_generated_answer_rules.txt";
		if (args.length < 6) {
			System.out.println("Insufficient commandline arguments: need quesiton_tree_file, questions_file, answers_file, questions_save_file, responses_save_file and rules_save_file in those order");
			System.exit(1);
		}
		// Print args
		System.out.println("Questions Tree File:" + args[0]);
		System.out.println("Questions File:" + args[1]);
		System.out.println("Answers File:" + args[2]);
		System.out.println("Questions Save File:" + args[3]);
		System.out.println("Responses Save File:" + args[4]);
		System.out.println("Rules Save File File:" + args[5]);
		// Initialize stuff from args
		String questions_tree_file = args[0];
		String questions_file = args[1];
		String answers_file = args[2];
		String questions_save_file = args[3];
		String generated_responses_file = args[4];
		String generated_rules_file = args[5];

		// String questions_tree_file = "natural_questions/natural_questions_dev_lexparsed_inline.txt";
		// String questions_file = "natural_questions/natural_questions_dev_questions.txt";
		// String answers_file = "natural_questions/natural_questions_dev_shortest_answers.txt";
		// String questions_save_file = "natural_questions/rule_based_system_natural_questions_saved_questions.txt";
		// String generated_responses_file = "natural_questions/rule_based_system_natural_questions_generated_answers.txt";

		// String questions_tree_file = "sample_questions/sample_questions_lexparsed_inline.txt";
		// String questions_file = "sample_questions/sample_questions.txt";
		// String answers_file = "sample_questions/sample_answers.txt";
		// String questions_save_file = "sample_questions/rule_based_system_sample_questions_saved_questions.txt";
		// String generated_responses_file = "sample_questions/rule_based_system_sample_questions_generated_answers.txt";

		// Squad Seq2Seq train set

		// String questions_tree_file = "../Data/squad_seq2seq_train/squad_train_lexparsed_q.txt";
		// String questions_tree_file = "../Data/squad_seq2seq_train/squad_train_lexparsed_q_from_lexparser_sh.txt";
		// String questions_file = "../Data/squad_seq2seq_train/squad_train_q.txt";
		// String answers_file = "../Data/squad_seq2seq_train/squad_train_a.txt";
		// // String questions_save_file = "squad_seq2seq_train/rule_based_system_squad_seq2seq_train_case_sensitive_saved_questions.txt";
		// // String generated_responses_file = "squad_seq2seq_train/rule_based_system_squad_seq2seq_train_case_sensitive_generated_answers.txt";
		// // String generated_rules_file = "squad_seq2seq_train/rule_based_system_squad_seq2seq_train_case_sensitive_generated_answer_rules.txt";
		// String questions_save_file = "squad_seq2seq_train/rule_based_system_squad_seq2seq_train_case_sensitive_saved_questions_lexparser_sh.txt";
		// String generated_responses_file = "squad_seq2seq_train/rule_based_system_squad_seq2seq_train_case_sensitive_generated_answers_lexparser_sh.txt";
		// String generated_rules_file = "squad_seq2seq_train/rule_based_system_squad_seq2seq_train_case_sensitive_generated_answer_rules_lexparser_sh.txt";

		// Final: Squad Seq2Seq dev test set
		// String questions_tree_file = "squad_seq2seq_dev_moses_tokenized/squad_seq2seq_dev_test_lexparsed_q_file.txt";
		// String questions_file = "squad_seq2seq_dev_moses_tokenized/squad_seq2seq_dev_test_q_file.txt";
		// String answers_file = "squad_seq2seq_dev_moses_tokenized/squad_seq2seq_dev_test_a_file.txt";
		// String questions_save_file = "squad_seq2seq_dev_moses_tokenized/rule_based_system_squad_seq2seq_dev_test_saved_questions.txt";
		// String generated_responses_file = "squad_seq2seq_dev_moses_tokenized/rule_based_system_squad_seq2seq_dev_test_generated_answers.txt";
		// String generated_rules_file = "squad_seq2seq_dev_moses_tokenized/rule_based_system_squad_seq2seq_dev_test_generated_answer_rules.txt";

		// Initialize the POS tagger
		MaxentTagger tagger = new MaxentTagger("stanford-postagger-2017-06-09/models/english-left3words-distsim.tagger");

		BufferedReader br_q_tree = new BufferedReader(new FileReader(questions_tree_file.trim()));
		BufferedReader br_q = new BufferedReader(new FileReader(questions_file.trim()));
		BufferedReader br_a = new BufferedReader(new FileReader(answers_file.trim()));

		BufferedWriter bw_q = new BufferedWriter(new FileWriter(questions_save_file));
		BufferedWriter bw_res = new BufferedWriter(new FileWriter(generated_responses_file));
		BufferedWriter bw_rule = new BufferedWriter(new FileWriter(generated_rules_file));
		try {
			String question_tree = br_q_tree.readLine().trim();
			question_tree = lowercase_tree_string(question_tree);
			String question_str = br_q.readLine().trim();
			String answer_str = br_a.readLine().trim();
			int missed_examples_count = 0;
			while (question_str != null) {
				// After reading the line which is a questions compute the answer

				// if answer str length is more than 5 words then we ignore it
				if(answer_str.split("\\s+").length > 6 || answer_str.split("\\s+").length == 0 || answer_str.isEmpty()) {
					question_tree = br_q_tree.readLine();
					question_tree = lowercase_tree_string(question_tree);
					question_str = br_q.readLine();
					answer_str = br_a.readLine();
					continue;
				}
				// if(!question_str.trim().equals("What problem can be caused by a player becoming out of alignment?")) {
				// 	question_tree = br_q_tree.readLine();
				// 	question_tree = lowercase_tree_string(question_tree);
				// 	question_str = br_q.readLine();
				// 	answer_str = br_a.readLine();
				// 	continue;
				// }
				if(!verify_tree_string(question_tree, question_str)) {
					System.out.println("QUESTION TREE: " + question_tree);
					System.out.println("QUESTION STR: " + question_str);
					missed_examples_count++;
				} else {
					// One process the verified examples
					System.out.println("Answer Length:" + answer_str.split("\\s+").length);
					Tree question = Tree.valueOf(question_tree);
					
					System.out.println(question);
					System.out.println(question_str);

					ArrayList<Pair> gen_responses_and_rules = question_to_answer(question, answer_str, tagger);
					
					// Save the files
					if(gen_responses_and_rules.size() > 0) {
						bw_q.write(question_str.trim() + "\n");
						for(Pair response_and_rule : gen_responses_and_rules) {
							String response = response_and_rule.response;
							String rule = response_and_rule.rule;
							bw_res.write(response.trim() + "\n");
							bw_rule.write(rule.trim() + "\n");
						}
						bw_res.write("\n");
						bw_rule.write("\n");
					}
				}

				question_tree = br_q_tree.readLine();
				question_tree = lowercase_tree_string(question_tree);
				question_str = br_q.readLine();
				answer_str = br_a.readLine();
			}
			System.out.println("Total missed examples:" + missed_examples_count);
			// String everything = sb.toString();
		} finally {
			br_q_tree.close();
			br_q.close();
			br_a.close();

			bw_q.close();
			bw_res.close();
			bw_rule.close();
		}
	}
}




