import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
from pymongo import MongoClient
import boto3

st.set_page_config(page_title='DHV Packages', layout='wide')

# First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# AWS S3 configuration
s3_bucket_name = 'healthpackage'
s3 = boto3.client('s3')

# Streamlit app
def main():
    st.title("DHV AI Startup Packages Search Demo")

    # Text input
    query = st.text_input("Enter Package Data")

    # Image search button
    if st.button("Search Package"):
        if query == "":
            st.error("Please enter the data you want to search")
        else:
            search(query)

def search(query, k=2):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)[0]  # Extract the tensor from the list

    # Reshape query embedding to match the expected input shape
    query_emb = query_emb.unsqueeze(0)  # Add a batch dimension

    # Then, we fetch the image embeddings from AWS S3
    img_names, img_emb = fetch_image_embeddings_from_s3()

    # Reshape image embeddings to match the expected input shape
    img_emb_reshaped = img_emb.view(img_emb.shape[0], -1)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb_reshaped, top_k=k)[0]  

    # Display the query image
    query_corpus_id = hits[0]['corpus_id']
    query_image_name = img_names[query_corpus_id]
    query_image_url = get_image_url_from_s3(query_image_name)
    query_image_link = get_image_link_from_s3(query_image_name)
    display_image(query_image_url, query_image_link, query_image_name)

    # Display related images
    for hit in hits[1:]:
        corpus_id = hit['corpus_id']
        image_name = img_names[corpus_id]
        image_url = get_image_url_from_s3(image_name)
        image_link = get_image_link_from_s3(image_name)
        display_image(image_url, image_link, image_name)

def fetch_image_embeddings_from_s3():
    # List all objects in the S3 bucket
    response = s3.list_objects_v2(Bucket=s3_bucket_name)
    objects = response['Contents']

    img_names = []
    img_emb = []
    for obj in objects:
        img_name = obj['Key']
        img_names.append(img_name)

        # Download the image file from S3
        s3.download_file(s3_bucket_name, img_name, f"./{img_name}")

        # Load and encode the image
        image = Image.open(f"./{img_name}")
        image_emb = model.encode(image, convert_to_tensor=True)
        img_emb.append(image_emb)

    return img_names, torch.stack(img_emb)

def display_image(image_url, image_link, image_name):
    if image_url:
        st.image(image_url, width=200)
    else:
        st.write("Image URL not found for", image_name)

    if image_link:
        st.write("Image Link:", image_link)
    else:
        st.write("Image Link not found for", image_name)

def get_image_url_from_s3(image_name):
    # Retrieve the image URL from AWS S3 based on the image name
    image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{image_name}"
    try:
        s3.head_object(Bucket=s3_bucket_name, Key=image_name)
        return image_url
    except:
        return None

def get_image_link_from_s3(image_name):
    # Connect to the MongoDB database
    client = MongoClient("mongodb://localhost:27017/")

    # Select the database and collection
    db = client["Kamthorn"]
    collection = db["packages"]

    # Retrieve the image link from the MongoDB database based on the image name
    document = collection.find_one({"source": image_name})
    if document and "link" in document:
        return document["link"]
    else:
        return None

if __name__ == "__main__":
    main()