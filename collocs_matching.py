import os, json
import spacy
from tqdm import tqdm
from typing import List, Union, Dict, Any, Tuple, Generator

class LexFunctionsMapper:
    def __init__(self, spacy_model: Union[str, None] = None, batch_size:int =32, n_process:int =5) -> None:
        """
        Mapper for Lex. Functions from corpus

        :param spacy_model: str, spacy model to use for lemmatization and syntax tree parsing
        :param batch_size: int, batch size for spacy pipeline
        :param n_process: int, num. of process to use for spacy pipeline
        """
        self.cc_contexts_dict = {} # output dict

        self.spacy_model = None
        if spacy_model:
            self.spacy_model = spacy.load(spacy_model)

        self.batch_size = batch_size
        self.n_process = n_process

    def get_sents(self, texts: List[str]) -> List[Any]:
        """
        Processing texts with spacy to get docs
        :param texts: list of str, texts to process
        :return: list of spacy.Doc, rocessed texts into docs
        """
        return list(self.spacy_model.pipe(texts, batch_size=self.batch_size, n_process=self.n_process))

    def get_indexes(self, doc: spacy.tokens.Doc, start_cc:str, end_cc:str) -> Union[Dict[str, Any], None]:
        """
        Retrieve indexes of lexical function. The function will check if there is any match of lemmas with provided collocation
        If there is, if the another part of collocation is in children dep. tree of the found part, it is a match and will be outputed
        :param doc: spacy.Doc, spacy doc to use for matching for the text
        :param start_cc: str, the first part of the collocation. For example, in collocation "pay rent" it is "pay"
        :param end_cc: str, the second part of the collocation. For example, in collocation "pay rent" it is "rent"
        :return: None if there is no match. Otherwise:
            dict: {
            start_cc_index": int, index of the first part of collocation in the token list of doc,
            "end_cc_index": int, index of the second part of collocation in the token list of doc,
            "tokens": list of str, list of tokens from doc,
            "lemmas": list of str, list of lemmas from doc
            }
        """

        lemmas = [x.lemma_ for x in doc]


        if start_cc in lemmas and end_cc in lemmas:
            for i, token in enumerate(doc):
                if token.lemma_ == start_cc:
                    children_tokens = list(token.children)
                    children_texts = [token.lemma_ for token in children_tokens]

                    if end_cc in children_texts:
                        end_index = children_texts.index(end_cc)
                        end_index = children_tokens[end_index].i
                        return {
                            "start_cc_index": i, "end_cc_index": end_index,
                            "tokens": [x.text for x in doc],
                            "lemmas": lemmas
                                }

        return None

    def save_cc_data(self, output: Any, save_dir: str, cc: str, cc_class: str) -> None:
        """Save the result into the save_dir/cc_class/{cc}.jsonl. The file will be updated with new samples"""
        os.makedirs(os.path.join(save_dir, str(cc_class)), exist_ok=True)

        with open(os.path.join(save_dir, str(cc_class), f"{cc}.jsonl"), "a") as f:
            for cc_data_dict in output:
                str_content = json.dumps(cc_data_dict)
                f.write(str_content)
                f.write("\n")

    def map_sentences(self, list_ccs: List[str], list_cc_classes: List[str], texts: List[str], save_dir: str) -> None:
        """
        Mapp sentences to all the lexical functions that they contain


        :param list_ccs: list of str. list of collocations to look for in the sentences. Should be split with " ". E.g.: "pay rent", "pay attention"
        :param list_cc_classes: list of str, corresponding classes to the collocations
        :param texts: list of str, texts to split into sentences to look into
        :param save_dir: str, saving dir to save mapping results
        :return: None, the results will be saved in the save_dir. For the format, see self.save_cc_data
        """

        docs = self.get_sents(texts)

        for cc, cc_class in zip(list_ccs, list_cc_classes):
            cc_sentences = []

            start_cc, end_cc = cc.split(" ")


            for doc in docs:
                for sentence_doc in doc.sents:
                    cc_indexes = self.get_indexes(
                        doc=sentence_doc, start_cc=start_cc, end_cc=end_cc
                    )
                    if cc_indexes:
                        cc_indexes["lf"] = cc
                        cc_sentences.append(cc_indexes)

            self.save_cc_data(
                output=cc_sentences,
                save_dir=save_dir,
                cc=cc,
                cc_class=cc_class
            )



def remove_txt_extension(filename: str) -> str:
    if filename.endswith(".txt"):
        return filename[:-len(".txt")]
    return filename

def read_ccs(dir_path: str) -> Tuple[List[str], List[str]]:
    """
    Parsing LFs

    :param dir_path: str, path to dir with the txt files. Each txt file contains a lexical function collocation on a new line
        The class  of LF is retrieved as a txt filename without the extencion
    :return: tuple:
        list of str: retrieved collocations
        list of str: corresponding classes names
    """
    texts = []
    classes = []

    for filename in os.listdir(dir_path):
        target_name = remove_txt_extension(filename)

        with open(os.path.join(dir_path, filename), "r") as f:
            lines = f.readlines()
            lines = [x.replace("\n", "") for x in lines]
            lines = [x for x in lines if x]
            classes += [target_name for _ in  range(len(lines))]
            texts += lines
    return texts, classes


def read_texts(json_list_path:str, n_texts_per_iter:int) -> Generator[List[str], None, None]:
    """
    Generating chunks of sentences from the provided texts
    :param json_list_path: str, path to json file with texts to parse
    :param n_texts_per_iter: int, batch size to process per time
    :return: generate a list of texts, batch of texts
    """
    texts = json.load(open(json_list_path, "r"))
    for ndx in range(0, len(texts), n_texts_per_iter):
        yield texts[ndx:min(ndx + n_texts_per_iter, len(texts))]


def map_lfs(args):

    ccs, ccs_classes = read_ccs(dir_path=args.lfs_hierarchy_dir)
    sentence_mapper = LexFunctionsMapper(
        spacy_model=args.spacy_model,
        batch_size=args.batch_size,
        n_process=args.n_process
    )

    for i, texts in tqdm(enumerate(read_texts(args.csv_path, n_texts_per_iter=args.n_texts_per_iter))):
        sentence_mapper.map_sentences(
            list_ccs=ccs,
            list_cc_classes=ccs_classes,
            texts=texts,
            save_dir=args.save_dir
        )
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='match sentences to LFs')

    parser.add_argument('--save_dir',
                        type=str,
                        default="syntaxTree_parsed_ccs_laz",
                        help='path to dir where parsed results will be stored')
    parser.add_argument('--csv_path',
                        type=str,
                        default="data/archive/data_larazon_publico_v2.csv",
                        help='path to dir the train/test split will be stored')
    parser.add_argument('--spacy_model', type=str,
                        default="spacy_models/es_core_news_sm-3.1.0/es_core_news_sm/es_core_news_sm-3.1.0",
                        help='Spacy model to use for syntax tree (ignored if use_spacy False)')
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help="batch size to use in spacy pipe")
    parser.add_argument('--n_process', type=int,
                        default=5,
                        help="n_process to use in spacy pipe")
    parser.add_argument('--n_texts_per_iter', type=int,
                        default=15,
                        help="number of texts to process per iteration")

    args = parser.parse_args()
    map_lfs(args)

