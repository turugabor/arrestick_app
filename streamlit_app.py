import streamlit as st
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import urllib
import biotite.structure.io.pdbx as pdbx
import io
import yaml
from yaml import CLoader as Loader

st.title('arreSTick')


@st.cache_data
def get_dicts():
    # load data for protein label dictionaries (antry - entry name conversions)
    uniprot = pd.read_csv('data/uniprot_entries.tsv', sep='\t')
    entries = uniprot.set_index("Entry Name")["Entry"].to_dict()
    return entries

@st.cache_data
def get_params():
    with open("model_params", "r") as f:
        params = yaml.load(f, Loader)
        
    aa_dict = params["model_1"]["aa_dict"]

    kernel = np.array(params["model_1"]["kernel_weights"])

    bias = params["model_1"]["conv_bias"]
    
    sigmoid_weight = params["model_1"]["sigmoid_weight"]
    sigmoid_bias = params["model_1"]["sigmoid_bias"]
    
    return aa_dict, kernel, bias, sigmoid_weight, sigmoid_bias


def sigmoid(x, sigmoid_weight, sigmoid_bias):
    return 1 / (1 + np.exp(-sigmoid_weight * x - sigmoid_bias))
    

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
        

def plot(seq, convoluted, confidence = None):
    
    data = pd.DataFrame([convoluted, list(seq), np.arange(len(seq))+1], index=["Convolutional value", "Amino acid", "Amino acid position"]).transpose()
    data["region"] = ""
    
    data["probability"] = data["Convolutional value"].apply(lambda x: sigmoid(x, sigmoid_weight, sigmoid_bias))
    
    ymin = 0
    ymax = 1
    
    for i, row in data.iterrows():
        data.loc[i, "region"] = f"{i+1}-{i+16} {seq[i:i+15]}"
        
        
    if confidence is not None:
        
        data['alphafold'] = [1 if conf>70 else 0 for conf in confidence]
        data["structure"] = ['>70' if conf>70 else '≤70' for conf in confidence]
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        
        plot = px.line(data_frame=data, y="probability", x = "Amino acid position", hover_data=["region"],
                    color_discrete_sequence=["black"])
        
        plot.update_layout(yaxis_range=[ymin, 1])
        
        plot.update_traces(customdata = data.region, hovertemplate = 'Position: %{x} <br>Probability: %{y:.2f} <br>Region: %{customdata} ')

        heatmap_range = np.arange(ymin, ymax)

        # create trace for the the alphafold structure heatmap
        trace1 = go.Heatmap(
                z=np.concatenate([data.alphafold.values.reshape(1,-1)]),
                x= data["Amino acid position"],
                y = [0,1],
                colorscale=["#ef8354", "#2d3142"],
                customdata=np.concatenate([data.structure.values.reshape(1,-1)]*heatmap_range.shape[0]),
                hovertemplate='Alphafold confidence: %{customdata}',
                name="",
                showscale=False,
                opacity=0.6)
        
        subfig.add_trace(trace1)
        
        
        
        subfig.add_traces(plot.data)
               
        
        subfig.layout.xaxis.title="Amino acid position"
        subfig.layout.yaxis.title="arreSTick sequence probability"
        subfig.layout.yaxis2.update(visible=False, showticklabels=False)

        # add legends
        subfig.add_shape(type="rect",
                x0=0, y0=1.05, x1=20, y1=1.10,
            line=dict(color="#ef8354"), fillcolor="#ef8354", opacity=0.6
            )
        
        subfig.add_annotation(x=25, y=1.08, xanchor="left",
            text="Alphafold confidence ≤ 70",
            showarrow=False,    
            )
        
        subfig.add_shape(type="rect",
                x0=0, y0=1.12, x1=20, y1=1.17,
            line=dict(color="#2d3142"), fillcolor="#2d3142", opacity=0.6
            )
        
        subfig.add_annotation(x=25, y=1.15, xanchor="left",
            text="Alphafold confidence > 70",
            showarrow=False,    
            )
        
        # set background to white
        subfig.update_layout(paper_bgcolor='white',plot_bgcolor='white')

        st.plotly_chart(subfig)
         
    else:
        plot = px.line(data_frame=data, y="probability", x = "Amino acid position", hover_data=["region"])
        plot.layout.yaxis.title="arreSTick sequence probability"
        
        plot.update_traces(customdata = data.region, hovertemplate = 'Position: %{x} <br>Probability: %{y:.2f} <br>Region: %{customdata} ')
        st.plotly_chart(plot)
    
entries = get_dicts()
aa_dict, kernel, bias, sigmoid_weight, sigmoid_bias = get_params()

with st.form("entry_form"):
    st.header("Convolute a protein")
    
    entry_name = st.text_input("Uniprot entry or entry name (e.g. V2R_HUMAN or P30518)", "V2R_HUMAN").upper()
    show_fasta = st.checkbox("Display FASTA sequence")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Convolute")
    if submitted:
            st.text("Move your cursor over the plot for additional information")
            try:
                seq, confidence = get_alphafold_data(entry_name)   
                convoluted = convolve(seq, kernel, aa_dict, bias)
                nans = np.empty(14)
                nans[:] = np.nan
                convoluted = np.append(convoluted, nans)
                
                plot(seq, convoluted, confidence)   
                 
                if show_fasta:
                    st.text("FASTA (sequence with max probability displayed with upper case)")
                    position = np.argmax(convoluted[:-15])
                    seq = "".join([x.lower() if ((i < position) | (i > position+14)) else x.upper() for i,x in enumerate(seq)])
                    
                    fasta = ""
                    for i in range(0,len(seq),50):
                        fasta += f"{i+1} \t {seq[i:i+50]}\n"
                    st.text(fasta)   
               
            except:
                st.write("invalid protein entry name or other error")  
    
                     
            

            
with st.form("seq_form"):
    st.header("Convolute custom protein sequence")
    seq = st.text_area('Enter protein sequence here', 
                       '''MLMASTTSAVPGHPSLPSLPSNSSQERPLDTRDPLLARAELALLSIVFVAVALSNGLVLAALARRGRRGHWAPIHVFIGHLCLADLAVALFQVLPQLAWKATDRFRGPDALCRAVKYLQMVGMYASSYMILAMTLDRHRAICRPMLAYRHGSGAHWNRPVLVAWAFSLLLSLPQLFIFAQRNVEGGSGVTDCWACFAEPWGRRTYVTWIALMVFVAPTLGIAACQVLIFREIHASLVPGPSERPGGRRRGRRTGSPGEGAHVSAAVAKTVRMTLVIVVVYVLCWAPFFLVQLWAAWDPEAPLEGAPFVLLMLLASLNSCTNPWIYASFSSSVSSELRSLLCCARGRTPPSLGPQDESCTTASSSLAKDTSS''').upper().replace("\n", "").replace(" ", "")
    
    submitted = st.form_submit_button("Convolute")
    if submitted:
        st.text("Move your cursor over the plot for additional information")
        #remove non amino acid characters
        seq_cleaned = "".join([a for a in seq if a in "QWERTIPASDFGHKLYCVNM"])
        
        if len(seq_cleaned) != len(seq):
            st.write("Non amino acid characters were removed")
            
        
            
        convoluted = convolve(seq_cleaned, kernel, aa_dict, bias)
        nans = np.empty(14)
        nans[:] = np.nan
        convoluted = np.append(convoluted, nans)
            
        plot(seq_cleaned, convoluted)


