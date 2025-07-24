import os
import re
import requests
import streamlit as st
from annotated_text import annotated_text # https://pypi.org/project/st-annotated-text/

SERVER = 'http://localhost:5004'
COLOR_MAP = {'cause':'#fea', 'enable':'#8ef', 'prevent':'#afa', 'intend':'#faf', 'invalid':'#faa'}
PRESETS = ['', 
        '[Wrong inferences example] Lalu , Rabri upbeat after success of shutdown 29th January 2010 01:40 PM An RJD activist wears a garland and crown made of vegetables and shouts slogans along with others during a protest against inflation in Patna . Protests were held across Andhra Pradesh criticising police action on Naidu and his supporters . Denied Aid , Dalit Boy tries to End Life. So we are asking people to come out because it may be the last time that we are going to have a peaceful and lawful protest in Hong Kong , ” said one of the organisers of the rally.  Mining for trouble Sino Gold Mining , which only last week announced a joint venture to expand exploration near its White Mountain Mine in Jilin province , had to halt operations yesterday as protesting farmers blocked the main access road .',
        
        '[Correct inferences example] At Balagangamanahalli panchayat in Dharmapuri , residents of Eechampatti village laid siege to the Nallampalli BDO s office in protest against non supply of water . 15th September 2015 05:49 AM THIRUVANANTHAPURAM : With the government appearing to be in no mood to meet the demand of the doctors of the health service , the Kerala Government Medical Officers Association spearheading the hunger strike in front of the state secretariat has called for intensifying the agitation in the coming days . Subcontractors  will  be offered a settlement and a swift transition to new management  is expected  to avert an exodus of skilled workers from Waertsilae Marines two big shipyards, government officials said. As part of the year-long partnership, Oral-B and Scientific American Custom Media are releasing a series of content, including educational resources from leading medical and dental researchers that will help readers better understand the connections between oral health and whole body health. NE Youth s Death Sparks Protest 01st February 2014 09:23 AM Nido Taniam , son of Arunachal Pradesh Congress legislator Nido Pavitra died on Thursday allegedly after being beaten up at a market area in Lajpar Nagar here .'
        ]

def app(available_models):
    # Title of the app
    st.set_page_config(page_title='Event Relation Extraction Demo')
    st.title("Event Relation Extraction Demo")
    # remove "Deploy button" - see https://discuss.streamlit.io/t/how-to-hide-or-remove-the-deploy-button-that-appears-at-the-top-right-corner-of-the-streamlit-app/55325
    st.markdown(r"""
    <style>
        .stDeployButton, .stAppDeployButton {
            visibility: hidden;
        }
    </style>
    
    <div>
        Code, documentation and paper at: <a href="https://github.com/ANR-kFLOW/Relation_extraction">https://github.com/ANR-kFLOW/Relation_extraction</a>
    </div>
    <hr>
    """, unsafe_allow_html=True)
    
    # Dropdown menu
    selected_st0 = st.selectbox("**Sentence Classification model** (filter out sentences with no event relations)", available_models['st0'])
    selected_st1 = st.selectbox("**Relation Classification model** between *cause*, *prevent*, *intend* and *enable*", available_models['st1'])
    selected_st2= st.selectbox("**Relation Extration model** (extract the word spans referring to events and relations)", available_models['st2'])
    
    user_text = st.selectbox("Text to analyse:", PRESETS, accept_new_options=True)
    user_text = re.sub(r"^\[.+\]", '', user_text, 0).strip()

    llm_api = st.text_input("OpenAI API key (only if you are using GPT-4)")
    
    # Button to trigger the output
    if st.button("Submit"):
        fail_flag = False
        
        params = {
            'st0': selected_st0,
            'st1': selected_st1,
            'st2': selected_st2,
            'q': user_text,
            'api': llm_api
        }

        if not user_text:
            st.write('Please submit a sentence or select a preset text')
            fail_flag = True

        if not llm_api and ('gpt4' in selected_st1 or 'gpt4' in selected_st2):
            st.write('You need an OpenAI key if you use GPT-4')
            fail_flag = True
            
        if fail_flag:
            return
        
        r = requests.get(SERVER+'/extract', params=params)
        result = r.json()

        if len(result) == 0:
            st.write('No entity relation found')
        else:
            for i in annotate_inf(result):
                annotated_text(i)

def annotate_inf(_raw_results):
    annotated_list = []
    test_list = []
    t_list = ['subj', 'obj']
    
    if len(_raw_results) == 0:
        pass
    elif 'roberta' in _raw_results[0]['st2_model']:
        for row in _raw_results:
            text = row['span_pred']
            rel = row['label']
            
            if not rel in COLOR_MAP.keys():
                label = 'invalid'
                rel = 'other(' + rel + ')'
            else:
                label = rel
            color = COLOR_MAP[label]
            
            subj_tag = ']'
            obj_tag = ']'
            
            subj_a = text.find('<ARG0>')
            subj_b = text.find('</ARG0>')
            
            if subj_a < subj_b:
                text = re.sub('<ARG0>', '[', text)
                text = re.sub('</ARG0>', subj_tag, text)
            else:
                text = re.sub('</ARG0>', '[', text)
                text = re.sub('<ARG0>', subj_tag, text)
            
            
            obj_a = text.find('<ARG1>')
            obj_b = text.find('</ARG1>')
            
            if obj_a < obj_b:
                text = re.sub('<ARG1>', '[', text)
                text = re.sub('</ARG1>', obj_tag, text)
            else:
                text = re.sub('</ARG1>', '[', text)
                text = re.sub('<ARG1>', obj_tag, text)
                
            text = re.sub(r"</?[^>]+>", "", text)
            
            subj_index = text.find(subj_tag)
            obj_index = text.find(obj_tag)
            
            if subj_index > obj_index:
                tag_order = ['obj', 'subj']
            else:
                tag_order = ['subj', 'obj']
            
            
            subj_tag2 = rel + '-subj'
            obj_tag2 = rel + '-obj'
            
            tag_list = {'subj':subj_tag2, 'obj':obj_tag2}
            h = highlight_sent(text, tag_order, tag_list)
            test_list.append(h)
            annotated_list.append(h)       
    else:
        for row in _raw_results:
            #you can potentially have it pull up the column name that has span_pred and use that as a variable
            sub_obj = row['span_pred']
            rel = row['label']
            if not rel in COLOR_MAP.keys():
                label = 'invalid'
                rel = 'other(' + rel + ')'
            else:
                label = rel
            color = COLOR_MAP[label]
            sentence = row['text']
            loop_list = [sentence]
            
            for i in range(2):
                temp_list = []
                obj = sub_obj[i]
                for entry in loop_list:
                    if isinstance(entry,str):
                        x = annotate_sub(entry, obj, rel + '-' + t_list[i], color)
                        temp_list.extend(x)
                    else:
                        temp_list.append(entry)
                loop_list = temp_list
            annotated_list.append(loop_list)
    return annotated_list
                
def annotate_sub(sent, obj, label, color):
    annotated_list = []
    if obj.lower() in sent.lower():
        parts = re.split(re.escape(obj), sent, flags=re.IGNORECASE, maxsplit=1)
        t = (obj, label, color)
        annotated_list.append(parts[0])
        annotated_list.append(t)
        annotated_list.append(parts[1])
        return annotated_list
    else:
        annotated_list.append(sent)
        return annotated_list

def extract_substring(text, start_tag='<ARG1>', end_tag='</ARG1>'):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    if start_index == -1 or end_index == -1:
        return None  # Tags not found

    return text[start_index:end_index]

def remove_tags(text):
    
    if text is None:
        return ''
    
    # Pattern to match anything between < and >
    cleaned_text = re.sub(r"</?[^>]+>", "", text)
    cleaned_text = re.sub('  ', ' ', cleaned_text)
    return cleaned_text

def highlight_sent(sent, tag_order, tag_list):
    tag = tag_list[tag_order[1]]
    rel_type = tag_list['subj'].replace('-subj', '')
    color = COLOR_MAP.get(rel_type, COLOR_MAP['invalid'])
    
    bracket_list = ['[', ']']
    split1 = sent.split('[', 1)
    split2 = split1[1].split(']', 1)
    first_split = [split1[0], split2[0], split2[1]]
    first_split_list = [first_split[0], ('[', '', color), first_split[1], (']', tag_list[tag_order[0]], color), first_split[2]]
    split_list = first_split_list
    
    for bracket in bracket_list:
        loop_list = []
        for entry in split_list:
            if isinstance(entry, str):
                loop_part = subsplit_text(entry, color, bracket, tag)
                loop_list.extend(loop_part)
            else:
                #print(entry)
                loop_list.append(entry)
        split_list = loop_list
        
    #print(split_list)
    #print('highlight_sent ends here')
    return(split_list)

def subsplit_text(part, color, s, tag):
    sub_list = []
    note = ''
    if s == ']':
        note = tag
    if s in part:
        parts = part.split(s)
        sub_list.append(parts[0])
        x = (s, note, color)
        sub_list.append(x)
        sub_list.append(parts[1])
    else:
        sub_list.append(part)
    return sub_list

def extract_spans(sents):
    span_list = []
    
    if not sents:
        return 0
    for sent in sents:
        loop_list = []
        # sometimes tags are swapped
        subj = extract_substring(sent, '<ARG0>', '</ARG0>') | extract_substring(sent, '</ARG0>', '<ARG0>')
        obj = extract_substring(sent, '<ARG1>', '</ARG1>') | extract_substring(sent, '</ARG1>', '<ARG1>')
        
        subj = remove_tags(subj)
        obj = remove_tags(obj)
        
        loop_list.append(subj)
        loop_list.append(obj)
        
        span_list.append(loop_list)
    return span_list

if __name__ == "__main__":
    SERVER = os.getenv('SERVER', SERVER)
    r = requests.get(SERVER+'/models')
    available_models = r.json()
    app(available_models)
    
