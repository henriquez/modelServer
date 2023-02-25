import chromadb
import pprint
import re

# initialize embedding model
client = chromadb.Client()
collection = client.create_collection("all-my-documents")
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
def get_paragraph_array(file='../tests/data/arubaWikipedia.txt'):
    with open(file) as f:
        source_documents = f.read()
    
    normalized_text = re.sub('\n{2,}', '\n\n', source_documents)
    paragraph_array = normalized_text.split('\n\n')
    array_cleaned = [paragraph for paragraph in paragraph_array if paragraph != '\n']

    # print(array_cleaned)
    return array_cleaned



'''
Top level flow
'''
paragraphs = get_paragraph_array()
# Chroma ids must be strings, just make up ids here in real life these
# should TODO: come from crawler and be associated with the paragraphs
id_list = [str(i) for i in range(len(paragraphs))]

# add documents to the embedding
collection.add(
    documents=paragraphs,
    ids=id_list,
)
print(f'Added {collection.count()} items to embedding.\n')


# request a result given a query
results = collection.query(
    query_texts=["where does aruba get its water and power?"],
    n_results=3
)


print('Query results:')
pp.pprint(results["documents"])




