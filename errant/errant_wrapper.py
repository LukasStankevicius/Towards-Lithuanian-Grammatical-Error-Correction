from collections import Counter

from errant.commands.compare_m2 import simplify_edits, process_edits, evaluate_edits, merge_dict, print_results
from errant.commands.parallel_to_m2 import parse_args, noop_edit
import errant
from contextlib import ExitStack


def parallel_to_m2(tuples, lev=False, merge="rules"):
    """

    :param tuples: list like [(original_sentence, corrected_sentence1, ...), ...]
    :param lev:
    :param merge: ["rules", "all-split", "all-merge", "all-equal"]
            "rules: Use a rule-based merging strategy (default)\n"
            "all-split: Merge nothing: MSSDI -> M, S, S, D, I\n"
            "all-merge: Merge adjacent non-matches: MSSDI -> M, SSDI\n"
            "all-equal: Merge adjacent same-type non-matches: MSSDI -> M, SS, D, I",
    :return: yields (original_multiline_string, corrected_multiline_string)
    """
    annotator = errant.load("lt")
    for orig, cors in tuples:
        output_str = ""
        orig = annotator.parse(orig)
        # Write orig to the output m2 file
        output_str += " ".join(["S"] + [token.text for token in orig]) + "\n"
        # Loop through the corrected texts
        for cor_id, cor in enumerate([cors]):  # here we imitate a list of corrections
            cor = cor.strip()
            # If the texts are the same, write a noop edit
            if orig.text.strip() == cor:
                output_str += noop_edit(cor_id) + "\n"
            # Otherwise, do extra processing
            else:
                # Parse cor with spacy
                cor = annotator.parse(cor)
                # Align the texts and extract and classify the edits
                edits = annotator.annotate(orig, cor, lev, merge)  # todo: looks like the main action is happening here
                # Loop through the edits
                for edit in edits:
                    # Write the edit to the output m2 file
                    output_str += edit.to_m2(cor_id) + "\n"
        # Write a newline when we have processed all corrections for each line
        yield output_str.strip()  # instead of ending with \n\n now we end with nothing


def tuples_to_scores(tuples):
    """

    :param tuples: list [("bad sentence", "hypothesis sentence", "reference sentence") , ...]
    :return: TP, FP, FN, precision, recall, f05
    """
    best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
    scores = []
    best_cats = {}
    kwargs = dict(lev=False, merge="rules")
    args = MyArgs()
    for sent_id, (hyp, ref) in enumerate(zip(parallel_to_m2([(t[0], t[1]) for t in tuples], **kwargs),
                                             parallel_to_m2([(t[0], t[2]) for t in tuples], **kwargs)
                                             )
                                         ):
        # Simplify the edits into lists of lists
        hyp_edits = simplify_edits(hyp)
        ref_edits = simplify_edits(ref)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, args)
        ref_dict = process_edits(ref_edits, args)
        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict, cat_dict = evaluate_edits(
            hyp_dict, ref_dict, best_dict, sent_id, args)
        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
        best_cats = merge_dict(best_cats, cat_dict)
        scores.append((count_dict["tp"], count_dict["fp"], count_dict["fn"]))
    return print_results(best_dict, best_cats, args), scores


class MyArgs:
    def __init__(self):
        self.beta = 0.5
        self.verbose = None
        self.dt = None
        self.ds = None
        self.cs = None
        self.cse = None
        self.single = None
        self.multi = None
        self.filt = None
        self.cat=None


if __name__ == "__main__":
    tuples = [
        ("žiūrėk, iš karto važiuoji ją knebinėti;", "žiūrėk, iš karto važiuoji ją knebinėti;", "žiūrėk, iš karto važiuoji ją knebinėti;"),
        ("Sakinys be kablelio.", "O čia, sakinys su kableliu", "O čia"),
        ("Vis tiek kol buvo galima, jei praeidavo", "Vistiek, kol buvo galima, jie būtinai pereidavo", "Vistiek, kol buvo galima, jie būtinai pereidavo"),
        ("Momka duokš valgyt", "Mama noriu valgyti", "Mama noriu valgyti"),
        ("Veiną kortą gyvens labiau turtingas karalius", "Vienas kartą gyveno labai turtingas karalius", "Vieną kartą gyveno labai turtingas karalius"),
        ("Čion yr lab gero sakinio pavyzdys .", "Čia būna labai geras sakinio pavyzdys .", "Čia yra labai geras sakinio pavyzdys .")
    ]
    print(tuples_to_scores(tuples))

