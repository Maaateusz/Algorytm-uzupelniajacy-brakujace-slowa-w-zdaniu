# stworzenie tablicy następnych słów
def create_tns(input_text, k):
    text_list = clean_tokenize_text(input_text)
    TNS = create_next_words_table(text_list, k)
    return TNS


import re
import string


# czyszczenie i tokenizacja tekstu wejściowego
def clean_tokenize_text(input_text):
    replace = [".", "!", "?", "-", ";", "—"]
    input_text = re.sub(r"[IVXLCDM]+\.\ ", " ", input_text)
    input_text = input_text.lower().replace("\n", " ")
    for repl in replace:
        input_text = input_text.replace(repl, " <eos> ")
    input_text = input_text.replace("\xa0", " ")
    input_text = input_text.translate(str.maketrans(string.punctuation, " " * 32))
    input_text = re.sub(r"[\W]", " ", input_text)
    input_text = re.sub(r"[\d]+", " <num> ", input_text)
    input_text = input_text.replace(" eos ", " <eos> ")
    input_text = input_text.replace(" mw ", " <mw> ")
    input_text = input_text.replace(" num ", " <num> ")  # NEW
    input_text = " ".join(input_text.split())
    input_text = re.sub(r"(\ <eos>)+", " <eos>", input_text)

    text_list = input_text.split()
    if text_list[0] != "<eos>":
        text_list = ["<eos>"] + text_list

    return text_list


# wygenerowanie tablicy następników
def create_next_words_table(text, k):
    text_list = []

    # <eos> k razy
    for word in text:
        if word == "<eos>":
            [text_list.append(word) for _ in range(0, k)]
        else:
            text_list.append(word)

    if k > 0:
        k_grams = create_n_gram(text_list, k)
        TNS = dict()

        for k_gram in set(k_grams):
            item = search_next_words(k_gram, text_list, k)
            TNS[k_gram] = item

        # posortowane według liczby następników
        TNS = dict(
            sorted(
                TNS.items(),
                key=lambda val: len(dict(val[1]).values()),
                reverse=True,
            )
        )
        return TNS
    else:
        return {}


# stworzenie n-gramu
def create_n_gram(input_text, n):
    temp = zip(*[input_text[i:] for i in range(0, n)])
    return [" ".join(el) for el in temp]


from collections import Counter

# wyszukiwanie następnych tokenów po k-gramie
def search_next_words(k_gram, text_list, k):
    text_len = len(text_list)
    words = list()

    k_gram_list = k_gram.split()
    for i in range(0, text_len):
        end_i = i + ((k + 1) - 1)
        sequence = text_list[i:end_i]
        if k_gram_list == sequence:
            if end_i >= text_len:
                words.append("<eos>")
            else:
                words.append(text_list[end_i])

    words_grouped = {}
    for word, count in sorted(
        Counter(words).items(), key=lambda item: item[1], reverse=True
    ):
        words_grouped[count] = (
            [word]
            if count not in words_grouped.keys()
            else words_grouped[count] + [word]
        )

    return words_grouped


# GraphViz needs to be installed!!!
from anytree import Node

# uzupełnienie zdania z brakami
def complete_missing_words(sentence, TNS, k):
    sentence_list = clean_tokenize_text(sentence)
    tree = Node("<eos>", occurrences=0)

    fill_prediction_tree(sentence_list, 1, k, tree, TNS)

    result_sentences = list()

    leafes = list()
    for leaf in tree.leaves:
        if len(leaf.path) == len(sentence_list):
            leafes.append(leaf)
    for leaf in leafes:
        path = leaf.path
        new_sentence = list()
        for node_in_path in path:
            new_sentence.append((node_in_path.name, node_in_path.occurrences))
        weight = sum_occurrences(new_sentence)
        result_sentences.append((weight, " ".join([word for word, _ in new_sentence])))

    return sorted(result_sentences, key=lambda item: item[0], reverse=True)


# wypełnienie drzewa z predykcjami dla zdania z brakami
def fill_prediction_tree(sentence: list, current_level: int, k: int, node: Node, TNS):
    if current_level < len(sentence):

        if sentence[current_level] == "<mw>":
            parents = node.ancestors + (node,)
            prev_k_nodes = parents[
                0 if (len(parents) - k) < 0 else (len(parents) - k) : len(parents)
            ]
            if current_level < k:
                old_nodes = prev_k_nodes
                prev_k_nodes = [Node("<eos>") for _ in range(k - len(old_nodes))] + [
                    nd for nd in old_nodes
                ]
            prev_k_words = [n.name for n in prev_k_nodes]
            prev_k_words_str = " ".join(prev_k_words)
            word_nexts = TNS.get(prev_k_words_str, {})
            for occurences_number in word_nexts.keys():
                for word in word_nexts.get(occurences_number):
                    if word != "<eos>":
                        new_node = Node(
                            f"{word}",
                            parent=node,
                            occurrences=occurences_number,
                        )
                        fill_prediction_tree(
                            sentence, current_level + 1, k, new_node, TNS
                        )

        elif (
            sentence[current_level] != "<mw>" and sentence[current_level - 1] == "<mw>"
        ):
            parents = node.ancestors + (node,)
            prev_k_nodes = parents[
                0 if (len(parents) - k) < 0 else (len(parents) - k) : len(parents)
            ]
            if current_level < k:
                old_nodes = prev_k_nodes
                prev_k_nodes = [Node("<eos>") for _ in range(k - len(old_nodes))] + [
                    nd for nd in old_nodes
                ]
            prev_k_words = [n.name for n in prev_k_nodes]
            prev_k_words_str = " ".join(prev_k_words)
            if sentence[current_level] in flatten(
                dict(TNS.get(prev_k_words_str, {})).values()
            ):
                new_node = Node(
                    f"{sentence[current_level]}", parent=node, occurrences=0
                )
                fill_prediction_tree(sentence, current_level + 1, k, new_node, TNS)

        else:
            new_node = Node(f"{sentence[current_level]}", parent=node, occurrences=0)
            fill_prediction_tree(sentence, current_level + 1, k, new_node, TNS)


def sum_occurrences(sentence):
    return sum(w[1] for w in sentence)


def flatten(L: list):
    return [item for sublist in L for item in sublist]
