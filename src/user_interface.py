import csv
import torch
import streamlit as st
from pathlib import Path
from train import MathEmbeddingModel
from data_loader import image_transform
from PIL import Image


BASE_DIR = Path(__file__).parent / '..'
DATA_DIR = BASE_DIR / 'data'
METADATA_PATH = DATA_DIR / 'pairs' / 'metadata.csv'
SECTION_DIR = DATA_DIR / 'sections'
CHECKPOINT_PATH = BASE_DIR / 'slow' / 'checkpoints_slow' / 'experiment_C_raised_weight' / 'best.pt'
EMBEDDING_PATH = DATA_DIR / 'embeddings' / 'embeddings_C_raised_weight.pt'
TOP_K = 5


@ st.cache_data
def load_metadata():
    metadata = {}
    with open(METADATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['section_id']] = row
    return metadata


@st.cache_data
def load_embeddings():
    return torch.load(EMBEDDING_PATH, weights_only=True, map_location='cpu')


@st.cache_resource
def load_model():
    model = MathEmbeddingModel(embedding_dim=128, unfreeze_layer4=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location='cpu'))
    model.eval()
    return model


def get_clusters(metadata):
    return sorted(set(content['cluster'] for content in metadata.values()))


def get_topics_for_cluster(metadata, cluster):
    return sorted(set(content['topic'] for content in metadata.values() if content['cluster'] == cluster))


def get_sections_for_filter(metadata, cluster=None, topic=None):
    results = []
    for id, content in metadata.items():
        if cluster is not None and content['cluster'] != cluster:
            continue
        if topic is not None and content['topic'] != topic:
            continue
        results.append(id)
    return results


def retrieve_by_image(query_image, model, embeddings, top_k=TOP_K):
    tensor = image_transform(query_image)
    with torch.no_grad():
        query_emb = model(tensor.unsqueeze(0)).squeeze(0)
    section_ids = list(embeddings.keys())
    emb_matrix = torch.stack(list(embeddings.values()))
    distances = (emb_matrix - query_emb.unsqueeze(0)).pow(2).sum(dim=1).sqrt()
    topk_indices = torch.argsort(distances)[:top_k]

    results = []
    for idx in topk_indices:
        results.append({'section_id': section_ids[idx], 'distance': distances[idx].item()})
    return results


def display(results, metadata):
    for result in results:
        section_id = result['section_id']
        content = metadata[section_id]
        image_path = SECTION_DIR / f"{section_id}.jpg"

        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(image_path)
            st.image(image, width='stretch')

        with col2:
            st.markdown(f'**Section ID: {section_id}**')
            st.write(f"Cluster: {content['cluster']}")
            st.write(f"Topic: {content['topic']}")
            if 'distance' in result:
                st.write(f"Distance: {result['distance']:.3f}")

        with st.expander(f'Click to enlarge {section_id}'):
            image = Image.open(image_path)
            st.image(image, width='stretch')

        st.divider()



def main():
    st.set_page_config(page_title="MathLens Retrieval System", layout="wide")
    st.title("MathLens - Section Retrieval System")
    st.write('Search handwritten math notes by similarity of uploaded image or by cluster/topic filters.')

    metadata = load_metadata()
    embeddings = load_embeddings()
    model = load_model()

    tab1, tab2 = st.tabs(["Image Search", "Text Search"])

    with tab1:
        st.header("Search by Image")
        st.write("Upload an image of a handwritten math note to find similar sections in the dataset.")
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption='Uploaded Image', width=400)

            with st.spinner('Retrieving similar sections...'):
                results = retrieve_by_image(query_image, model, embeddings)

            st.subheader(f"Top {TOP_K} Similar Sections")
            display(results, metadata)

    with tab2:
        st.header("Search by Cluster and or Topic")
        clusters = get_clusters(metadata)
        selected_cluster = st.selectbox("Select Cluster", ["All"] + clusters)

        if selected_cluster != "All":
            topics = get_topics_for_cluster(metadata, selected_cluster)
            selected_topic = st.selectbox("Select Topic", ["All"] + topics)
        else:
            selected_topic = "All"

        if st.button("Search"):
            if selected_cluster == "All":
                filtered_sections = get_sections_for_filter(metadata)
            elif selected_topic == "All":
                filtered_sections = get_sections_for_filter(metadata, cluster=selected_cluster)
            else:
                filtered_sections = get_sections_for_filter(metadata, cluster=selected_cluster, topic=selected_topic)

            st.subheader(f"Found {len(filtered_sections)} Sections")
            if filtered_sections:
                results = [{'section_id': section_id} for section_id in filtered_sections]
                display(results, metadata)
            else:
                st.info('No sections found for the selected filters.')



if __name__ == "__main__":
    main()