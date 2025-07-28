import os
import yaml
import configparser

CACHE_DIR = 'out/'
DEFAULT_PORT = 5004
SWAGGER_URL = '/swagger'
SWAGGER_TEMPLATE = 'static/swagger-template.yaml'
SWAGGER_OUT = 'static/swagger.yaml'  # URL for exposing Swagger file
AVAILABLE_LLMS = ['zephyr', 'dpo', 'una', 'solar', 'gpt4']
AVAILABLE_MODELS = {}
DATA_PATH = 'data/'

import werkzeug
from flask import Flask, render_template, jsonify, request
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint

from pipeline import run_pipeline

def model_names_in_dirs(directory_list):
    options = []
    for directory_path in directory_list:
        options += model_names_in_dir(directory_path)
    return options

def model_names_in_dir(directory_path):
    options = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isdir(file_path):
            prefix = filename + '_'
            for modelname in os.listdir(file_path):
                if os.path.isdir(file_path):
                    option = prefix + modelname
                    options.append(option)
    return options

def populate_available_models():
    AVAILABLE_MODELS['st0'] = model_names_in_dir('pretrained_models/st0')

    AVAILABLE_MODELS['st1']= model_names_in_dirs(['pretrained_models/st1','pretrained_models/st3'])+ AVAILABLE_LLMS + ['off']

    AVAILABLE_MODELS['st2']= model_names_in_dirs(['pretrained_models/st2','pretrained_models/st3'])+ AVAILABLE_LLMS + ['off']

    AVAILABLE_MODELS['f'] = [filename
            for filename in os.listdir(DATA_PATH) 
            if os.path.isfile(os.path.join (DATA_PATH, filename))]

# Populate options for users in the swagger config
def set_choices(yaml_file):
    with open(SWAGGER_TEMPLATE, 'r') as stream:
        yconfig = yaml.load(stream, Loader=yaml.CLoader)

    parameters = yconfig['paths']['/extract']['get']['parameters']
    params = {}
    for x in parameters:
        params[x['name']] = x
    
    for x in ['st0','st1','st2', 'f']:
        params[x]['enum'] = AVAILABLE_MODELS[x]

    print('Loading Swagger with the following configuration')
    print(yconfig)
    with open(SWAGGER_OUT, 'w') as file:
        yaml.dump(yconfig, file)
    return yconfig

# Collects all of the responses from the user and puts it into one dict
def get_params():
    params = request.args
    response = {}
    
    for key in params.keys():
        value = params.get(key)
        response[key] = value
    print('Extractor called with the following parameters')
    print(response)
    return response

#this generates the preset labels for each of the subtasks
#it is important to note that the preset may not exist but that is checked in pipeline.py
def check_cache(cache_list):
    skip_path = {}
    f = cache_list['f']
    st0 = 'tf-' + f + '-' + 'filter-' + cache_list['filter']
    st1 = ''
    st2 = ''
    tasks = ['st0_preset', 'st1_preset', 'st2_preset']
    
    if 'st1' in cache_list:
        st1 = st0 + '-' + 'st1-' + cache_list['st1']
        
    if 'st2' in cache_list:
        st2 = st0 + '-' + 'st2-' + cache_list['st2']
        
    directory_path = CACHE_DIR
    skip_path['st0_preset'] = directory_path + st0 + '.csv'
    skip_path['st1_preset'] = directory_path + st1 + '.csv'
    skip_path['st2_preset'] = directory_path + st2 + '.csv'
    
    return skip_path

# Reads the user arguments and creates a config file based on that
def set_config(flags):
    config = configparser.ConfigParser()
    config['TEMP'] = {}
    ctemp = config['TEMP']
    preset_labels = {}
    ctemp['preset_cache_dir'] = CACHE_DIR
    
    check_preset_flag = False
    
    if 'f' not in flags and 'q' not in flags:
        print('Missing both `f` and `q`')
        raise BadParameters()

    if 'q' not in flags:
        ctemp['test_file'] = 'new_data/' + flags['f']
        check_preset_flag = True
        preset_labels['f'] = flags['f']
    
    filter_s = flags['st0']
    
    filter_parts = filter_s.split('_')
    filter_prefix = filter_parts[0] + '_' + filter_parts[1]
    filter_name = filter_parts[2]
    for i in range(3,len(filter_parts)):
        filter_name = filter_name + '_' + filter_parts[i]
    filter_path = 'pretrained_models/st0/' + filter_prefix + '/' + filter_name
    ctemp['filter_model_path'] = filter_path
    
    preset_labels['filter'] = 'roberta-' + filter_name
    
    ctemp['rebel_flag'] = 'False'
    
    if flags['st1'] != 'off':
        s = flags['st1']
        if s == 'zephyr':
            ctemp['LLMS_llm'] = 'zephyr'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st1'] = 'llm-zephyr'
            ctemp['llm_st1_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st1_mod'] = 'zephyr'
        elif s == 'dpo':
            ctemp['LLMS_llm'] = 'dpo'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st1'] = 'llm-dpo'
            ctemp['llm_st1_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st1_mod'] = 'dpo'
        elif s == 'una':
            ctemp['LLMS_llm'] = 'una'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st1'] = 'llm-una'
            ctemp['llm_st1_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st1_mod'] = 'una'
        elif s == 'solar':
            ctemp['LLMS_llm'] = 'solar'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st1'] = 'llm-solar'
            ctemp['llm_st1_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st1_mod'] = 'solar'
        elif s == 'gpt4':
            ctemp['LLMS_llm'] = 'gpt4'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st1'] = 'llm-gpt4'
            ctemp['llm_st1_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st1_mod'] = 'gpt4'
        else:
            parts = s.split('_')
            prefix = parts[0] + '_' + parts[1]
            model_name = parts[2]
            if parts[1] != 'st3':
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st1/' + prefix + '/' + model_name
                ctemp['st1_model_name_or_path'] = path
                ctemp['subtask1_flag'] = 'True'
                ctemp['st1_roberta_flag'] = 'True'
                preset_labels['st1'] = 'roberta-' + model_name
            else:
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st3/' + prefix + '/' + model_name
                ctemp['rebel_st1_flag'] = 'True'
                ctemp['rebel_flag'] = 'True'
                ctemp['subtask3_flag'] = 'True'
                ctemp['st1_roberta_flag'] = 'False'
                preset_labels['st1'] = 'rebel-' + model_name
                ctemp['rebel_st1_mod'] = path
    
    
    if flags['st2'] != 'off':
        s = flags['st2']
        if s == 'zephyr':
            ctemp['LLMS_llm'] = 'zephyr'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st2'] = 'llm-zephyr'
            ctemp['llm_st2_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st2_mod'] = 'zephyr'
        elif s == 'dpo':
            ctemp['LLMS_llm'] = 'dpo'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st2'] = 'llm-dpo'
            ctemp['llm_st2_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st2_mod'] = 'dpo'
        elif s == 'una':
            ctemp['LLMS_llm'] = 'una'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st2'] = 'llm-una'
            ctemp['llm_st2_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st2_mod'] = 'una'
        elif s == 'solar':
            ctemp['LLMS_llm'] = 'solar'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st2'] = 'llm-solar'
            ctemp['llm_st2_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st2_mod'] = 'solar'
        elif s == 'gpt4':
            ctemp['LLMS_llm'] = 'gpt4'
            ctemp['LLM_flag'] = 'True'
            preset_labels['st2'] = 'llm-gpt4'
            ctemp['llm_st2_flag'] = 'True'
            ctemp['subtask3_flag'] = 'True'
            ctemp['llm_st2_mod'] = 'gpt4'
        else:
            parts = s.split('_')
            prefix = parts[0] + '_' + parts[1]
            model_name = parts[2]
            if parts[1] != 'st3':
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st2/' + prefix + '/' + model_name
                ctemp['st2_pretrained_path'] = path
                ctemp['st2_load_checkpoint_for_test'] = path + '/pytorch_model.bin'
                ctemp['subtask2_flag'] = 'True'
                ctemp['st2_roberta_flag'] = 'True'
                preset_labels['st2'] = 'roberta-' + model_name
            else:
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st3/' + prefix + '/' + model_name
                ctemp['rebel_flag'] = 'True'
                ctemp['rebel_st2_flag'] = 'True'
                ctemp['subtask3_flag'] = 'True'
                preset_labels['st2'] = 'rebel-' + model_name
                ctemp['rebel_st2_mod'] = path
    
    if 'q' in flags:
        ctemp['text_from_user'] = flags['q']
        check_preset_flag = False
    if 'api' in flags:
        ctemp['LLMS_api_key'] = flags['api']
    if check_preset_flag:
        print(preset_labels)
        cache_dict = check_cache(preset_labels)
        for key in cache_dict.keys():
            ctemp[key] = cache_dict[key]
    
    with open('config_swagger.cfg', 'w') as configfile:
        config.write(configfile)
    return config

app = Flask(__name__)
api = Api(app)

# Swagger UI setup
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI endpoint
    '/'+SWAGGER_OUT,  # Swagger file URL
    config={  # Swagger UI config overrides
        'app_name': "Relation Detection API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

class BadParameters(werkzeug.exceptions.HTTPException):
    code = 400
    description = 'At least one parameter between `f` (dataset) or `q` (sentence) is required'

@app.errorhandler(werkzeug.exceptions.BadRequest)
def handler(e):
    return e.description, e.code

app.register_error_handler(BadParameters, handler)

class extract(Resource):
    def get(self):
        # Extract query parameters with default values
        flags = get_params()
        config = set_config(flags)

        json = run_pipeline(config)
        return jsonify(json)
api.add_resource(extract, '/extract')

class models(Resource):
    def get(self):
        return jsonify(AVAILABLE_MODELS)
api.add_resource(models, '/models')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_pipe():
    json = run_pipeline('config_swagger.cfg')
    return json

if __name__ == "__main__":
    populate_available_models()
    set_choices(SWAGGER_OUT)
    app.run(port=DEFAULT_PORT, host='0.0.0.0', debug=False)