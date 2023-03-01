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


############### Classes ################


class Slot:
    '''Slot constructor mirrors same in dialog.js, enabling the client to pass it to
    the server and vice versa. Use *exactly* the same variable name as BotConfig.js
    This is used for expert-authored Q&A conversations
    '''
    def __init__(self, name, type = '', ask = '', replyId = '', trigger = ''):
        self.name = name
        self.type = type
        self.ask = ask
        self.replyId = replyId
        self.trigger = trigger



class Round:
    '''Round constructor for storing one conversation round, defined as starting with 
    a user query/reply, 
    and ending with the answer this service replied with. Includes the embedding
    model and LLM portions of the reply. Sent to the web-client where its used in
    the reply to the user. This constructor mirrors same named attrs *exactly* in dialog.js, 
    enabling the client to pass it to the server and vice versa. 
    TODO: add the top 4 new attrs to dialog.js. 
    '''

    EMBED_RESULT_COUNT = 3 # number of results returned by embedding model

    def __init__(self, preamble, userInput, 
                slot = None, 
                frameId = None, 
                replyValues = None, 
                replyIndexes = None, 
                ending = None):
        ##### these vars are passed in from the Conversation object
        self.preamble = preamble

        ###### these vars are passed in from the web-client
        # free text user input given *this round*, including the first user query.
        # used when replyValues are not used.
        # TODO: have this class deserialize JSON to set? 
        self.userInput = userInput
        self.slot = slot
        self.frameId = frameId
        # user selections for when free text user input is not allowed
        self.replyValues = replyValues
        self.replyIndexes = replyIndexes
        self.ending = ending

        ##### server side calculated values #####

        # After Round is created, populate the server source values - as a 
        # result, Round is ready to be returned to the user after created.
        # list of strings, where each string is one embedding model result
        self.embeddingResults = self.get_embeddings()
        # reply returned to the user (from the LLM)
        self.answer = self.get_llm_summary()
        # embedding model provides links we show to the user. list items are
        # string hrefs
        # TODO: self.referenceLinks = ?
        # TODO: who serializes the class to JSON? a class method or fastAPI?

    
    def prompt(self) -> str:
        '''Return prompt to pass to the LLM, including the user query, embeddings, and
        other fixed strings to make the query better
        '''
        prompt = (f"{self.preamble}\n\n" 
                  f"Context:\n{self.embeddingResults}\n\n---\n\n"
                  f"Question: {self.userInput}\nAnswer:")
        
        return prompt


    def get_embeddings(self) -> list:
        '''Set self.embeddingResults with the embedding model. No return value.
        Called in constructor
        '''
        # request a result from embedding 
        result_dict = collection.query(query_texts=[self.userInput], 
                                       n_results=Round.EMBED_RESULT_COUNT)
        return result_dict['documents']



    def get_llm_summary(self) -> str:
        '''Returns response from LLM (A summary aka 'completion' from OpenAI 
        See https://platform.openai.com/docs/api-reference/completions/create 
        '''
        MODEL = "text-davinci-003"
        # Constants used in LLM API call:
        # max tokens to generate in the completion
        MAX_TOKENS = 100
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 
        # will make the output more random, while lower values like 0.2 will make it 
        # more focused and deterministic.
        TEMPERATURE = 0
        # Number between -2.0 and 2.0. Positive values penalize new tokens based on 
        # whether they appear in the text so far, increasing the model's likelihood 
        # to talk about new topics.
        PRESENCE_PENALTY = 0
        # Number between -2.0 and 2.0. Positive values penalize new tokens based on 
        # their existing frequency in the text so far, decreasing the model's 
        # likelihood to repeat the same line verbatim.
        FREQUENCY_PENALTY = 0
        # Number of completions to generate
        N = 1
 

        try:
            # Create a completions using the question and context
            prompt = self.prompt()
            
            response = openai.Completion.create(
                prompt=prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                model=MODEL,
                presence_penalty=PRESENCE_PENALTY,
                frequency_penalty=FREQUENCY_PENALTY,
                n=N
            )
            return response["choices"][0]["text"].strip()

        except Exception as e:
            print(e)
            return ""



class Conversation:
    '''Conversation constructor for storing and operating on conversation state over a
    single-topic conversation where history needs to be stored. This includes
    the prompt passed to the LLM, user inputs/queries, and context pulled
    from the embedding model. LT this should mirror the slot, frame, and round objects
    in dialog.js and botConfig.js. They need a string free text reply object added.
    TODO: determine if the missing attrs in dialog.js are needed server side.
    '''
    def __init__(self):
        # list of Round objects
        # TODO: at conversation start, does webclient pass in completedRounds?
        self.completedRounds = [] 
        # preambles begin the prompt string
        self.preamble = ("Acting as an expert on Aruba, answer the "
            "question based on the context below. If the question can't be "
            "answered based on the context, say \"I don't know\"\n\n")
        
    


############### Utility functions ################



def get_source_documents(file: str ='../tests/data/arubaWikipedia.txt') -> list:
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
    with open(file) as f:
        source_documents = f.read()
    
    normalized_text = re.sub('\n{2,}', '\n\n', source_documents)
    paragraph_array = normalized_text.split('\n\n')
    array_cleaned = [paragraph for paragraph in paragraph_array if paragraph != '\n']

    # print(array_cleaned)
    return array_cleaned







def init_embeddings(paragraphs: list) -> None:
    '''Initialize the embedding model with source text.
    Returns: None
    '''

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








############ Top level flow ##############

paragraphs = get_source_documents()
init_embeddings(paragraphs)

############ Conversation starts #############
# TODO: server receives user query from web client

conversation = Conversation()
round1 = Round(conversation.preamble, 'Who first settled Aruba?') 
conversation.completedRounds.append(round1)
print('-------------- DEbug -------------')
# print(f'Embedding results:\n{round1.embeddingResults}\n')
print(f'Prompt:\n{round1.prompt()}\n')
print(f'Answer:\n{round1.answer}\n')


'''
# Round 2
embedding_list = get_embeddings(f'{user_query}' )
print(f'Embedding results:\n{embedding_list}\n')

prompt = make_prompt(user_query, embedding_list)
print(f'Prompt:\n{prompt}\n')

answer = get_llm_summary(prompt)
print(f'Answer:\n{answer}\n')
'''











