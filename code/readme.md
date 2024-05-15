
## Dataset Download
First download the full 82GB vimeo septuplet 90K dataset from [http://toflow.csail.mit.edu/](http://toflow.csail.mit.edu/).

Extract the contents of the zip and place into the `model` folder. The directory should be named `vimeo_septuplet`, and it should contain a `sequences` subdirectory. 

## Commands
Train the model:
- Command: ``python DNSRNet.py train <model_name> <training_set_path>`` 

Evaluate the model's accuracy:
- Command: ``python DNSRNet.py eval <model_name> <evaluation_set_path>``

## Benchmarks
Sample script to run benchmark schemes have included.
- Command: ``python benchmark_fastdvdnet.py --test_path <test_path> --noise_sigma <noise_sigma> --save_path <save_path> --model_file <model_file> --suffix {}``