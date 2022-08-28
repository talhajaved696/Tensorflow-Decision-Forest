import tensorflow_decision_forests as tfdf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Decision Forest Deep Learning App',
    layout='wide')

#---------------------------------#
# Model building

## Spliting Dataset
def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

def build_model(df):
    
    if uploaded_file:
        label = df.columns[-1] # Selecting the last column as Label
    else:
        label = "species"
    # Data splitting
    classes = df[label].unique().tolist()
    df[label] = df[label].map(classes.index)

    st.markdown('**Data splits**')
    
    train_ds_pd, test_ds_pd = split_dataset(df)
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)
    st.write('Training and Test set')
    st.info("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))
    

    st.markdown('**Variable details**:')
    st.write('X variable')
    st.info(list(df.columns))
    st.write('Y variable')
    st.info(label)

    # Specify the model.
    if option == 'RandomForest':
        model_1 = tfdf.keras.RandomForestModel(verbose=2)
    if option == 'GradientBoost':
        model_1 =  tfdf.keras.GradientBoostedTreesModel(verbose=2)
        
    # Train the model.
    model_1.fit(x=train_ds)

    st.subheader('Model Performance')
    model_1.compile(metrics=["accuracy"])
    evaluation = model_1.evaluate(test_ds, return_dict=True)

    st.markdown('**Test set**')
    st.write('Evaluation: Loss/Accuracy')
    
    for name, value in evaluation.items():
        st.info(f"{name}: {value:.4f}")


    st.subheader('Model Details / Variable Importance')
    st.json(model_1.make_inspector().variable_importances())

    st.subheader('Plotting Results')
    logs = model_1.make_inspector().training_logs()

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    st.pyplot(fig)

    st.subheader("Model Plot")
    components.html(tfdf.model_plotter.plot_model(model_1, tree_idx=0, max_depth=3), height=400)


#---------------------------------#
st.write("""
# Tensorflow Deep Learning App - Decision Forests 🌳🌳🌳
In this implementation, the TensorFlow Decision Forests (Tf-Df) is used to build a classification model.
Try using different algorithms!
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header(' 📁 1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("⚠️ Make sure Label is the last column in file")
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')

st.subheader('Model Selection')
option = st.radio(
        'Select Classification Model?',
        ('RandomForest', 'GradientBoost'))
    
st.write('Model:', option)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    build_model(df)
else:
    st.info('📌 Awaiting for CSV file to be uploaded')
    if st.button('Click to use Example Dataset'):
        
        df = pd.read_csv('./penguins.csv')

        st.markdown('The Penguin dataset is used as the example.')
        st.markdown("https://allisonhorst.github.io/palmerpenguins/articles/intro.html")
        st.write(df.head(5))

        build_model(df)