# Pipeline and Demo

The pipeline addresses the following sub-tasks:

- `st0` filters out the entries that have no relationship
- `st1` classifies the type of relationship in a given entry among `cause`, `enable`, `intend`, or `prevent`
- `st2` identify the subject and the object of a given relationship for each entry
- `st3` these models output a combination of `st1` and `st2` (kept for legacy)

In the options, `st3` models can be used to do either st1 or st2 tasks.
However, when the model is called both tasks are done regardless if the pipeline requests for only one.

## Import the pretrained models

Create and populate the `pretrained_models` folder.
The hierarchy for storing the pretrained models should look like

```yaml
pretrained_models:
  st0:
    roberta_st0:
      pretrained_model.pt
  st1:
    roberta_st1:
      pretrained_model_folder
  st2:
    roberta_st2:
      pretrained_model_folder
  st3:
    rebel_st3:
      pretrained_model.pth
```

When adding new files into `st0` or `rebel_st3` be mindful of keeping the file as the same file type as shown in the example.

TBW where to find these files

## Use the Docker image

Build the docker image using

```cli
docker build -f Dockerfile.api -t kflow/rel_extraction .
```

then run the image with

```cli
docker run -d -p 5004:5004 -v $(pwd)/pretrained_models:/pretrained_models -v $(pwd)/out:/out --name kflow_rel_extraction kflow/rel_extraction
```


[//]: # (docker run -d -p 5002:5004 -v /data/kflow-dataset:/data -v /data/kflow-model/pretrained_models:/pretrained_models -v /data/kflow-model/out:/out --name kflow_rel_extraction kflow/rel_extraction)


When running the link that is given in the terminal change the local port number to the number that you exported from your local machine.
For the previous example it would be http://127.0.0.1:5002/swagger/
> (Be sure to add the /swagger to the end of the link)

For the demo, run

```cli
docker build -f Dockerfile.demo -t kflow/rel_extraction_demo .
```

then run the image with

```cli
docker run -d -p 5003:5004 --name kflow_demo kflow/rel_extraction_demo
```

## Running the pipeline manually

There are two options for running the pipeline without the use of the web application: passing the arguments to call_pipeline.py, and creating a config file which is passed to call_pipeline.py

Arguments available:

- `--test_file`   Input the path to the file you want to use for inferences
    
- `--st1_mod`   Input the path to the pretrained model you are using for st1. Be sure to include the path in the pretrained models folder
    
- `--st2_mod`   Input the path to the pretrained model you are using for st2.  

- `--config_path`   If there is a config file available input the path and that will be taken as a priority over the other inputs. 
  
- `--text_from_user`   Type out your sentences in quote marks and be sure to separate them with periods. This will be used instead of the test file
  
- `--skip_st1`   Type in true if you do not want the pipeline to perform st1
  
- `--skip_st2`   Type in true if you do not want the pipeline to perform st2
  
Any value that is not specified will be filled in by the default value that is described in config_default.cfg this means that the pipeline can be ran with a command as simple as:

    python pypeline.simplified.py
    
Arguments can be added at the end to change the configuration of the pipeline, like:

    python pypeline.simplified.py --text_from_user "This is a test sentence."

Which would make the pipeline use the given sentence instead of the default csv file

When using a config file do a command like:

    python pypeline.simplified.py --config_file /path/to/file

For the config file the format looks like: 

    [TEMP]
    Argument_A = x
    Argument_B = y
      
A full example of this can be found in example.cfg

Inputting arguments through a config file works the same way as typing it out in a command line.

---

## Citation

If you use this software, please cite ([bib file](https://hal.science/hal-05135516v1/bibtex)):

Gustavo Flores Miguel, Youssra Rebboud, Pasquale Lisena, Raphäel Troncy.
**Streamlining Event Relation Extraction: A Pipeline Leveraging Pretrained and Large Language Models for Inference.**
In: *EKAW 2024, 24th International Conference on Knowledge Engineering and Knowledge Management*, Poster and Demo Track, CEUR, Nov 2024, Amsterdam, Netherlands.
https://ceur-ws.org/Vol-3967/PD_paper_184.pdf
