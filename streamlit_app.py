import streamlit as st
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import urllib
import biotite.structure.io.pdbx as pdbx
import io


st.title('arreSTick')

@st.cache_data
def get_dicts():
    # load data for protein label dictionaries (antry - entry name conversions)
    uniprot = pd.read_csv('data/uniprot_entries.tsv', sep='\t')
    entries = uniprot.set_index("Entry Name")["Entry"].to_dict()
    return entries

@st.cache_data
def get_params():
    amino_acids = "QWERTIPASDFGHKLYCVNM"
    embeddings = [-2.1997268, -1.1420017, -0.37575176, -2.1997268, 1.3075894, -2.1997268, -1.1420017, -2.1997268, 1.3075894, -0.37575176, -0.37575176, -1.1420017, -1.1420017, -2.1997268, -1.1420017, -1.1420017, -1.1420017, -0.37575176, -1.1420017, -0.37575176]
    aa_dict = dict(zip(list(amino_acids), embeddings))

    kernel = np.array([1.3060381 , 0.5402229 , 0.36126667, 1.6820629 , 0.37439126,
        0.83508617, 0.9705989 , 0.6321398 , 0.5831004 , 0.21672164,
        1.3454887 , 0.5097931 , 0.77933246, 0.2553734 , 1.1463245 ])

    bias = -0.2149748
    
    return amino_acids, embeddings, aa_dict, kernel, bias


def convolve(seq, kernel, aa_dict, bias):
    seq_translated = np.array(list(map(lambda x: aa_dict[x], seq)))
    convoluted = np.convolve(seq_translated, kernel[::-1], mode='valid')+bias
    
    return convoluted

def get_alphafold_data(entry_name):
        """downloads alphafold .cif file for the input protein and extracts sequence and model confidence array for each amino acid position

        Args:
            entry_name (str): uniprot entry name of the protein

        Returns:
            tuple of sequence and confidence array: the protein sequence from the alphafold website with the model confidence array. 
                            If the entry is not available on the server, "x" insted of the sequence and an array of [0] as the confidence is returned
        """
        try:
            if "_" in entry_name:
                entry = entries[entry_name]
                
            else:
                entry = entry_name
            
            connection = urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v4.cif")
            databytes = connection.read()
            connection.close()
            cif_txt = databytes.decode("utf8")

            f = io.StringIO(cif_txt)
            cif = pdbx.PDBxFile.read(f)
            
            confidence = pd.DataFrame(cif.get_category("ma_qa_metric_local")).metric_value.astype(float).values
            sequence = cif.get_category("entity_poly")["pdbx_seq_one_letter_code"]

            return sequence, confidence
        
        except:
            pass

def plot(seq, confidence = None):
    
    convoluted = convolve(seq, kernel, aa_dict, bias)
    nans = np.empty(14)
    nans[:] = np.nan
    convoluted = np.append(convoluted, nans)

    data = pd.DataFrame([convoluted, list(seq), np.arange(len(seq))+1], index=["Convolutional value", "Amino acid", "Amino acid position"]).transpose()
    data["region"] = ""
    for i, row in data.iterrows():
        data.loc[i, "region"] = f"{i+1}:{seq[i:i+15]}"
        
    
    
    if confidence is not None:
        data['alphafold'] = [1 if conf>70 else 0 for conf in confidence]
        data["structure"] = ['>70' if conf>70 else '≤70' for conf in confidence]
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig1 = px.line(data_frame=data, y="Convolutional value", x = "Amino acid position", hover_data=["region"],
                    color_discrete_sequence=["grey"])
        fig2 = px.imshow(data.alphafold.values.reshape(1,-1), height=50)

        fig2.update_traces(dict(showscale=False, 
                            coloraxis=None), selector={'type':'heatmap'})

        fig2.update(data=[{'customdata':data.structure.values.reshape(1,-1), 'hovertemplate': 'Alphafold score: %{customdata}'}])

        subfig.add_traces(fig1.data + fig2.data)
        subfig.layout.xaxis.title="Amino acid position"
        subfig.layout.yaxis.title="Convolutional value"
        subfig.layout.yaxis2.update(visible=False, showticklabels=False)
        subfig.layout.yaxis.title="Convolutional value"

    
        st.plotly_chart(subfig)
         
    else:
        fig = px.line(data_frame=data, y="Convolutional value", x = "Amino acid position", hover_data=["region"])
        st.plotly_chart(fig)
    
    
entries = get_dicts()
amino_acids, embeddings, aa_dict, kernel, bias = get_params()

with st.form("entry_form"):
   entry_name = st.text_input("Uniprot entry or entry name (e.g. V2R_HUMAN or P30518)", "V2R_HUMAN")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Convolute")
   if submitted:
        try:
            seq, confidence = get_alphafold_data(entry_name)
       
            plot(seq, confidence)
        except:
            st.write("invalid protein name or other error")   



with st.form("seq_form"):
    seq = st.text_area('Replace protein sequence', '''
    MLMASTTSAVPGHPSLPSLPSNSSQERPLDTRDPLLARAELALLSIVFVAVALSNGLVLA
    ALARRGRRGHWAPIHVFIGHLCLADLAVALFQVLPQLAWKATDRFRGPDALCRAVKYLQM
    VGMYASSYMILAMTLDRHRAICRPMLAYRHGSGAHWNRPVLVAWAFSLLLSLPQLFIFAQ
    RNVEGGSGVTDCWACFAEPWGRRTYVTWIALMVFVAPTLGIAACQVLIFREIHASLVPGP
    SERPGGRRRGRRTGSPGEGAHVSAAVAKTVRMTLVIVVVYVLCWAPFFLVQLWAAWDPEA
    PLEGAPFVLLMLLASLNSCTNPWIYASFSSSVSSELRSLLCCARGRTPPSLGPQDESCTT
    ASSSLAKDTSS''').replace("\n", "").replace(" ", "")
    submitted = st.form_submit_button("Convolute")
    if submitted:
        plot(seq)


