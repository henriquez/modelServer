# To run: `cd modelserver` then `python embedding.py`
# Debugging "module not found": kill terminal, recreate, run `poetry shell`
import chromadb
import pprint
import re
import os
import openai

# initialize embedding model
client = chromadb.Client()
collection = client.create_collection("all-my-documents")

# get GPT3 LLM key
# for usage stats see https://platform.openai.com/account/usage this is on
# logan@henriquez.net account
openai.api_key = os.environ["OPENAI_API_KEY"]
# the below gets rid of the warning on run
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pp = pprint.PrettyPrinter(indent=4)


'''
Utility functions
'''
# imports text from source_documents and clean it
# Returns: array of strings, each is one paragraph from source document
# Issues: Leaves in some non-text chars, and topic titles are not combined with 
# the paragraph below them.
# Features to add: when processing html, need page URLs and ideally paragraph
# level URLs paired with the text, so it will return two arrays, as that's what the
# Chroma collection.add function wants.
# TODO: when crawler is implemented, this should return a Page object, that includes
# text and URL as attributes. Then when answers are given for a query, can 
# use the URL as a citation.
# Debug: the file path is relative to the current directory where the python
# command was run, not where this file is.
# for PDF see https://github.com/pdfminer/pdfminer.six
def get_source_documents(file: str ='../tests/data/arubaWikipedia.txt') -> list:
    with open(file) as f:
        source_documents = f.read()
    
    normalized_text = re.sub('\n{2,}', '\n\n', source_documents)
    paragraph_array = normalized_text.split('\n\n')
    array_cleaned = [paragraph for paragraph in paragraph_array if paragraph != '\n']

    # print(array_cleaned)
    return array_cleaned


'''
Request summary aka 'completion' from OpenAI completions endpoint (LLM)
See https://platform.openai.com/docs/api-reference/completions/create for params
Returns: str with the response from the LLM
'''
def get_llm_summary(
    # encode as str, array of str, array of tokens, or array of token arrays
    # should include everything in the prompt including embeddings and user query
    prompt: str,
    model: str = "text-davinci-003",
    # max tokens to generate in the completion
    max_tokens: int = 100,
    # What sampling temperature to use, between 0 and 2. Higher values like 0.8 
    # will make the output more random, while lower values like 0.2 will make it 
    # more focused and deterministic.
    temperature: float = 0,
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on 
    # whether they appear in the text so far, increasing the model's likelihood 
    # to talk about new topics.
    presence_penalty: float = 0,
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on 
    # their existing frequency in the text so far, decreasing the model's 
    # likelihood to repeat the same line verbatim.
    frequency_penalty: float = 0,
    # Number of completions to generate
    n: int = 1,
    debug: bool = False

) -> str:

    if debug:
        print("Prompt:\n" + prompt)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    

'''
Slot constructor mirrors same in dialog.js, enabling the client to pass it to
the server and vice versa. Use *exactly* the same variable name as BotConfig.js
This is used for expert-authored Q&A conversations
'''
class Slot:
    def __init__(self):
        self.name:      str = ''
        self.type:      str = ''
        self.ask:       str = ''
        self.replyId:   str = ''
        self.trigger:   str = ''


'''
Round constructor for storing one conversation round, defined as starting with 
a user query/reply, 
and ending with the answer this service replied with. Includes the embedding
model and LLM portions of the reply. Sent to the web-client where its used in
the reply to the user. This constructor mirrors same in dialog.js, enabling 
the client to pass it to the server and vice versa. TODO: add the top 4 new 
attrs to dialog.js
'''
class Round:
    def __init__(self):
        # free text user input for when replyValues are not used
        self.user_reply: str = ''
        # list of strings, where each string is one embedding model result
        self.embedding_results: list = []
        # what was shown to the user from the LLM
        self.summarized_answer: str = ''
        # embedding model provides links we show to the user. list items are
        # string hrefs
        self.reference_links: list = [] 
        self.slot: Slot = None
        self.frameId: str = ''
        # user selections for when free text user input is not allowed
        self.replyValues: list = []
        self.replyIndexes: list = []
        self.ending: str = ''
        self.stats: str = ''


'''
Conversation constructor for storing and operating on conversation state over a
single-topic conversation where history needs to be stored. This includes
the prompt passed to the LLM, user inputs/queries, and context pulled
from the embedding model. LT this should mirror the slot, frame, and round objects
in dialog.js and botConfig.js. They need a string free text reply object added.
'''
class Conversation:
    def __init__(self):
        # list of Round objects
        self.completedRounds: list = [Round] 
        # preambles begin the prompt string
        self.preamble: str = ("Acting as an expert on Aruba, answer the "
            "question based on the context below. If the question can't be "
            "answered based on the context, say \"I don't know\"\n\n")
        


'''
Build the prompt to pass to the LLM, including the user query, embeddings, and
other fixed strings to make the query better
Returns: str with the prompt
'''
def make_prompt(
    user_query: str,
    embedding_results: list = [],

) -> str:
    
    prompt = ("Context: {embedding_results}\n\n---\n\n" 
    f"Question: {user_query}\nAnswer:")

    return prompt



'''
Initialize the embedding model with source text.
Returns: None
'''
def init_embeddings(
        paragraphs: list
    ) -> None:
    # Chroma ids must be strings, just make up ids here in real life these
    # should TODO: come from crawler and be associated with the paragraphs in a Page
    # object/class. For embedding bots, use a URI (URI = scheme ":" ["//" authority] path ["?" query] ["#" fragment])
    # as metadata like "https://page.support/bots/{uuid}". Theoreticaly the bot could be 
    # hosted anywhere like the company's website, this gives flexibility when the 
    # user clicks the link to launch the bot. In web-client code, look for the bots
    # word in the path then grab the UUID and serve the right bot.
    id_list = [str(i) for i in range(len(paragraphs))]

    # add documents to the embedding
    # TODO: add URLs as metadatas when adding paragraphs. For bots, also use
    # a URL - see above.
    collection.add(
        documents=paragraphs,
        ids=id_list,
    )
    print(f'Added {collection.count()} items to embedding.\n')



'''
Query the embedding model given a user query.
Returns: list of strings containing paragraphs from the model
'''
def get_embeddings(
    query: str
) -> list:
    # request a result from embedding 
    result_dict = collection.query(
        query_texts=[query],
        n_results=3
    )

    return(result_dict['documents'])



'''
Top level flow
'''
paragraphs = get_source_documents()
init_embeddings(paragraphs)
user_query = 'Who first settled Aruba?' 

# Round 1
embedding_list = get_embeddings(user_query)
print(f'Embedding results:\n{embedding_list}\n')

prompt = make_prompt(user_query, embedding_list)
print(f'Prompt:\n{prompt}\n')

answer = get_llm_summary(prompt)
print(f'Answer:\n{answer}\n')

# Round 2
embedding_list = get_embeddings(f'{user_query}' )
print(f'Embedding results:\n{embedding_list}\n')

prompt = make_prompt(user_query, embedding_list)
print(f'Prompt:\n{prompt}\n')

answer = get_llm_summary(prompt)
print(f'Answer:\n{answer}\n')











