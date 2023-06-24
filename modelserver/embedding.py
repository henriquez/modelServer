# To run: `cd modelserver` then `python embedding.py`
# Debugging "module not found": kill terminal, recreate, run `poetry shell`
import chromadb
import pprint
import re
import os
import openai
from dotenv import load_dotenv

load_dotenv() # loads OPENAPI key from ./.env


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
        self.preamble = ("Acting as an expert on Aruba, answer the questions below"
                         "based on the context below. If a question can't be "
                         "answered based on the context, say \"I don't know\".")
        



class Slot:

    def __init__(self, 
                 name: str, 
                 type: str, 
                 ask: str, 
                 replyId: str, 
                 trigger = str) -> None:
        
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

    def __init__(self, 
                conversation: Conversation, 
                userInput: str, 
                slot: Slot = None, 
                frameId: str = None, 
                replyValues: list = None, 
                replyIndexes: list = None, 
                ending: str = None) -> None:


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
        # We save everything in a Round that is needed to re-create the prompt
        # with history in the next conversation round.
        # list of strings, where each string is one embedding model result
        # TODO: should embeddings be a object with referenceLinks and
        # text?

        # save current embedding (list of strings) in round obj
        self.embeddings = self.get_embeddings(self.userInput)

        # assemble prompt, including previous rounds
        request = self.assemble_llm_request(self.userInput, self.embeddings,
                                            conversation)
        
        # set answer in this round (reply returned to the user from the LLM)
        self.answer = self.make_llm_request(request)

        # TODO: who serializes the class to JSON? a class method or fastAPI?


    
    def assemble_llm_request(self, userInput: str, embeddings,
                                conversation: Conversation) -> str:
        '''Return prompt to pass to OpenAI ChatCompletions endpoing.
        Including the user query, embeddings, and
        other fixed strings to make the query better. There is 
        no memory of past requests, so include all past requests and responses.
        4096 tokens is limit for gpt-3.5-turbo. Truncate at 2000 chars to be safe.
        See https://platform.openai.com/docs/guides/chat/introduction
        '''
        # preamble only needed once in initial system role
        messages = [{"role": "system", "content": conversation.preamble}]      

        # add previous round user questions, embeddings and bot answers
        for round in conversation.completedRounds:
           messages.append({"role": "user", "content": 
                            f"{(' ').join(round.embeddings)}\n{round.userInput}" 
                            })
           messages.append({"role": "assistant", "content": round.answer})

        # add this round's question and embeddings
        messages.append({"role": "user", "content": 
                         f"{(' ').join(embeddings)}\n{userInput}" 
                        })
        
        print('-------------- BEGIN messages -------------')
        print(messages)
        print('-------------- END messages -------------')

        return messages


    def get_embeddings(self, userInput: str) -> list:
        
        EMBED_RESULT_COUNT = 3
        # request a result from embedding 
        result_dict = collection.query(query_texts=[userInput], 
                                       n_results=EMBED_RESULT_COUNT)
        #p = pprint.PrettyPrinter()
        #p.pprint(result_dict['documents'][0])
        # for unknown reason, documents is a list with one element, inside
        # of which is a list of str where each str is one embedding result
        # up to EMBED_RESULT_COUNT
        return result_dict['documents'][0]



    def make_llm_request(self, 
                        # list of messages as defined by https://platform.openai.com/docs/guides/chat/introduction
                        messages: list) -> str:
        '''Returns string response from chat completions endpoint '''
        MODEL = "gpt-3.5-turbo"
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
            response = openai.ChatCompletion.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                presence_penalty=PRESENCE_PENALTY,
                frequency_penalty=FREQUENCY_PENALTY,
                n=N,
                messages=messages
            )
            # see https://platform.openai.com/docs/guides/chat/response-format
            return response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(e)
            return ""



    


############### Utility functions ################



def get_source_documents(file: str ='../tests/data/arubaWikipedia.txt') -> list:

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
    # as metadata like "https://example/bots/{uuid}". Theoreticaly the bot could be 
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
round1 = Round(conversation, 'Who first colonized Aruba?') 
conversation.completedRounds.append(round1)


print('-------------- ANSWER 1-------------')
print(f'Answer:\n{round1.answer}\n')

round2 = Round(conversation, 'Who colonized the island next?') 
conversation.completedRounds.append(round2)

print('-------------- ANSWER 2-------------')
print(f'Answer:\n{round2.answer}\n')












