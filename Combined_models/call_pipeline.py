import argparse
import configparser
import os
from two_step_model import run_pipeline
import pandas as pd
from datetime import datetime

available_llms = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "una": "fblgit/UNA-TheBeagle-7b-v1",
    "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
    "gpt4": "OpenAI-GPT4"  # Added GPT-4
}

# load the default configuration
default_config = configparser.ConfigParser()
default_config.read('config_default.cfg')
default_dict = {section: dict(default_config.items(section)) for section in default_config.sections()}


#this will return a dict with the preset labels for all of the presets used
def build_preset_label(t_f, filter, st1, st2, preset_labels):
    presets_dict = {}
    
    #this gets the name of the test file
    tf = t_f.split('/')[-1]
    
    #this gets the name of the pretrained model
    filter_name = filter.split('/')[-1]
    
    #this saves the path of the filter preset
    filter_preset = 'out/tf-' + tf + '-filter-' + preset_labels['filter'] + filter_name + '.csv'
    presets_dict['st0_preset'] = filter_preset
    
    #this allows st1 and st2 preset to use the filter label as part of their preset
    #this is important because dataset and model used for filtering could change the output which requires a different preset
    filter_label = 'tf-' + tf + '-filter-' + preset_labels['filter'] + filter_name
    
    #each of these checks are if the subtask is part of the pipeline
    if st1 != 'None':
        st1_name = st1.split('/')[-1]
        st1_preset = 'out/' + filter_label + '-st1-' + preset_labels['st1'] + st1_name + '.csv'
        presets_dict['st1_preset'] = st1_preset
        
    if st2 != 'None':
        st2_name = st2.split('/')[-1]
        st2_preset = 'out/' + filter_label + '-st2-' + preset_labels['st2'] + st2_name + '.csv'
        presets_dict['st2_preset'] = st2_preset
        
    return presets_dict

#the expectation is that it will receive a dict that contains the vital information to build a config file
def create_config(build_dict):
    
    llms = ["zephyr", "dpo", "una", "solar", "gpt4"]
    
    #config is the final config that will be passed to the pipeline
    config = configparser.ConfigParser()
    config['TEMP'] = {}
    
    #this keeps track of which type of models are being used for a given subtask
    preset_labels = {}
    
    #this keeps track of whether user text is used or not
    ut = False
    
    #if text doesn't have .csv it is most likely user submitted text
    if '.csv' in build_dict['text']:
        config['TEMP']['test_file'] = build_dict['text']
    else:
        ut = True
        config['TEMP']['text_from_user'] = build_dict['text']
    
    #these are basic arguements that can just be passed
    config['TEMP']['preset_cache_dir'] = 'out/'
    config['TEMP']['filter_threshold'] = str(build_dict['filter_threshold'])
    config['TEMP']['filter_model_path'] = build_dict['filter_mod']
    config['TEMP']['llms_api_key'] = str(build_dict['llms_api_key'])
    
    #this part would have to change if there are more types of files for filter than roberta
    preset_labels['filter'] = 'roberta-'
    
    #this is to override the preset that already exists
    config['TEMP']['override_preset'] = build_dict['override_preset']
    
    #if the skip is selected none of these arguements need to be in the config file
    if build_dict['skip_st1'] == 'False':
        #each of these blocks assign the appropriate flags required for each type of model used for the subtask
        if 'roberta' in build_dict['st1_mod']:
            config['TEMP']['subtask1_flag'] = 'True'
            config['TEMP']['st1_model_name_or_path'] = build_dict['st1_mod']
            config['TEMP']['st1_roberta_flag'] = 'True'
            preset_labels['st1'] = 'roberta-'
            
        if 'rebel' in build_dict['st1_mod']:
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['rebel_st1_mod'] = build_dict['st1_mod']
            config['TEMP']['rebel_flag'] = 'True'
            config['TEMP']['rebel_st1_flag'] = 'True'
            preset_labels['st1'] = 'rebel-'
        
        if any(mod in build_dict['st1_mod'] for mod in llms):
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['llm_st1_mod'] = build_dict['st1_mod']
            config['TEMP']['llm_flag'] = 'True'
            config['TEMP']['llm_st1_flag'] = 'True'
    
    #this does the same as the previous block except for st2
    if build_dict['skip_st2'] == 'False':
            
        if 'roberta' in build_dict['st2_mod']:
            config['TEMP']['subtask2_flag'] = 'True'
            config['TEMP']['st2_pretrained_path'] = build_dict['st2_mod']
            config['TEMP']['st2_load_checkpoint_for_test'] = build_dict['st2_mod'] + '/pytorch_model.bin'
            config['TEMP']['st2_roberta_flag'] = 'True'
            preset_labels['st2'] = 'roberta-'
            
        if 'rebel' in build_dict['st2_mod']:
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['rebel_st2_mod'] = build_dict['st2_mod']
            config['TEMP']['rebel_flag'] = 'True'
            config['TEMP']['rebel_st2_flag'] = 'True'
            preset_labels['st2'] = 'rebel-'
        
        if any(mod in build_dict['st2_mod'] for mod in llms):
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['llm_st2_mod'] = build_dict['st2_mod']
            config['TEMP']['llm_flag'] = 'True'
            config['TEMP']['llm_st2_flag'] = 'True'
            
    #presets are not needed when the user provides text
    if not ut:
        #the presets are now added to the config file
        presets_dict = build_preset_label(build_dict['text'], build_dict['filter_mod'], build_dict['st1_mod'], build_dict['st2_mod'], preset_labels)
        config['TEMP'].update(presets_dict)
    
    return config
    



def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default='None', 
        help="A csv file containing the test data."
    )
    parser.add_argument('--filter_mod', type=str, default='None', help='Path to the model used to filter the data')
    parser.add_argument('--filter_threshold', type=float, default=0.8, help='Threshold for classification in the filter')
    parser.add_argument(
        "--st1_mod",
        type=str,
        default='None',
        help="Model used for st1"
    )
    parser.add_argument(
        "--st2_mod",
        type=str,
        default='None',
        help="Model used for st2",
    )
    
    parser.add_argument('--llms_api_key', help='API key for GPT-4', required=False)
    parser.add_argument(
        "--config_path",
        default='None',
        help="This is the path to the config file",
    )
    parser.add_argument('--text_from_user', default='None', required=False)
    
    parser.add_argument('--skip_st1', default='False', help='This tells the pipeline to not perform st1', required=False)
    parser.add_argument('--skip_st2', default='False', help='This tells the pipeline to not perform st2', required=False)
    parser.add_argument('--override_preset', default='False', required=False)
    args = parser.parse_args()
    return args


def main_call(param=None, flag=False):
    args = parse_args()
    if flag:
        args.text_from_user = param['q']
        args.filter_mod = param['filter']
        args.st1_mod = param['st1']
        args.st2_mod = param['st2']
        
        args.llms_api_key = param['api']
    
    default_used = False
    user_dict = {}
    arg_list = [args.test_file, args.filter_mod, args.st1_mod, args.st2_mod, args.text_from_user]
    
    #checks if the user submitted any arguments that would change the config file from the default
    if all(x == arg_list[0] for x in arg_list):
        user_submitted = False
    else:
        user_submitted = True
    if args.skip_st1 != 'False' or args.skip_st2 != 'False':
        user_submitted = True
        

    #this is check is for the case when a user submits a config file
    if args.config_path != 'None':
        #this part updates the current arguments to the ones defined in the config file
        config = configparser.ConfigParser()
        config.read(args.config_path)
        for key in config['TEMP']:
                value = config['TEMP'].get(key)
                if hasattr(args, key):
                    attr_type = type(value)
                    setattr(args, key, attr_type(config['TEMP'][key]))
            
        user_submitted = True # in order to trigger next if

    if user_submitted:
        #this defines the dictionary representing the inputs from the user and filling in None when not provided
        build_dict = vars(args)      
        build_dict['text'] = args.test_file
        
        #These checks make changes based on whether a subtask is being skipped
        if build_dict['skip_st1'] != 'False':
            build_dict['st1_mod'] = 'None'
            
        if build_dict['skip_st2'] != 'False':
            build_dict['st2_mod'] = 'None'
        
        #this checks if the user provided their own text and if they didn't the csv file is used
        if args.text_from_user == 'None':
            if build_dict['text'] == 'None':
                build_dict['text'] = default_dict['TEMP']['test_file']
        else:
            build_dict['text'] = args.text_from_user
            
        #these checks fill in the parts that the user did not specify with the default parameters

        def_dict = default_dict['TEMP']

        if build_dict['filter_mod'] == 'None':
            build_dict['filter_mod'] = def_dict['filter_mod']
            
        if build_dict['st1_mod'] == 'None' and build_dict['skip_st1'] == 'False':
            build_dict['st1_mod'] = def_dict['st1_mod']
            
        if build_dict['st2_mod'] == 'None' and build_dict['skip_st2'] == 'False':
            build_dict['st2_mod'] = def_dict['st2_mod']
         
        if build_dict['st1_mod'] == 'gpt-4' or build_dict['st2_mod'] == 'gpt-4':
            if build_dict['llms_api_key'] == 'None':
                print('You need to give an OpenAI api key if you are going to prompt gpt-4')
                return 0
            
    else:
        #this is the default if nothing has been provided by user
        build_dict = default_dict['TEMP'].copy()
        
        build_dict['text'] = build_dict['test_file']
        build_dict['filter_threshold'] = args.filter_threshold
        build_dict['llms_api_key'] = args.llms_api_key
        build_dict['skip_st1'] = args.skip_st1
        build_dict['skip_st2'] = args.skip_st2
        build_dict['override_preset'] = args.override_preset

    #this creates the config file from the build dict which contains all of the necessary information to make a config file
    config = create_config(build_dict)
    with open('config_temp.cfg', 'w') as configfile:
        config.write(configfile)

    #now the newly created config file will be used    
    json_dict = run_pipeline(config)
    
    # if there isn't any detected relations 
    if len(json_dict) == 0:
        print('There are no causal sentences')
    else:
        df = pd.DataFrame(json_dict)
        os.makedirs('out', exist_ok = True)
        df.to_csv('out/'f'final-combined_pred-{datetime.now()}.csv')
    
    return json_dict
    
if __name__ == "__main__":
    main_call()