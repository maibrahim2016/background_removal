### Cloning the repository
---
First clone the repository:

    git clone <repository url>

Then **determine which specific version of the microservice is required** (refer to the owner of this module, or the person who requested for the deployment), and check out the corresponding branch:

    git checkout <branch>


### Using virtualenv
---
This section only concerns you if you want to use [`virtualenv`](https://virtualenv.pypa.io/en/latest/). If you don't wish to use a virtual environment, you can skip this part. **Be warned that not using `virtualenv` might result in some really painful dependency management though, so it's not advisable.**

If you have an existing virtual environment you wish to use, activate it:
    
    source /path/to/virtualenv_dir/bin/activate

or alternatively, create a new virtual environment (make sure `virtualenv` is already installed, obviously):

    virtualenv -p /path/to/python/executable /path/to/virtualenv_dir


### Getting sepyroth
---
First, check which version is required in [**requirements.txt**](requirements.txt), then look for a corresponding release branch in the `sepyroth` repository i.e. if the requirements file says `sepyroth==0.2.1`, there should be an `r0.2.1` branch. **If you cannot find the corresponding release branch, please check with the owner of this repository.**

Once you have determined which branch of `sepyroth` you require, clone the required version from [**here**](https://bitbucket.org/123rf-data/sepyroth):

    git clone <sepyroth url> --branch <required branch> --single-branch --depth 1

Install `sepyroth's` requirements:

    pip install -r /path/to/sepyroth/requirements.txt

Then install `sepyroth` itself:

    python /path/to/sepyroth/setup.py install


### External dependencies
---
Depending on the underlying OS, you may or may not run into an error that looks something like

    Traceback (most recent call last):
        File "/var/project/segmentation/segmentation/src/segmenter_microservice.py", line 9, in <module>
            import cv2
        File "/var/project/segmentation/lib/python3.6/site-packages/cv2/__init__.py", line 3, in <module>
            from .cv2 import *
    ImportError: libSM.so.6: cannot open shared object file: No such file or directory

If you do encounter this error, you can try installing one or more of the following libraries, which you're probably missing:

    yum install -y libXext libSM libXrender


### Installing microservice requirements
---
Next, install the microservice's requirements:

    pip install -r /path/to/this/repo/requirements.txt


### Configuring the microservice
---
You'll have to create a set of configurations for the microservice. An example JSON configuration file is provided [here](src/configs/segmenter.json.example), so start by making a copy then re-configuring it as necessary.

Generally speaking, the configurations that you'll need to change are:

1. `model_name` : name of the model, set at model server startup
2. `model_server_host` : hostname of the model server
3. `model_server_port` : port that the model server is configured to listen to for RPC requests
4. `pidfile` : where you would like to store the pidfile of the service
5. `stdout` : where you would like the standard output stream to be written to (set to /dev/null or similar if you don't want to preserve this)
6. `stderr` : where you would like the standard error stream to be written to (set to /dev/null or similar if you don't want to preserve this)
7. `logging.handlers.info.filename` : where you would like info logs to be written to (set to /dev/null or similar if you don't want to preserve this)
8. `logging.handlers.error` : where you would like error logs to be written to (set to /dev/null or similar if you don't want to preserve this)

Refer to [this](src/configs/segmenter.json.desc) for a more comprehensive description of the available configurations.


### Creating the service
---
To facilitate things for the engineers, we've established an agreement to create our microservices as system-level services, so that they can be managed with `systemctl`.

To begin creating the service, start by making a copy of the service template file provided [here](src/configs/segmenter.service.example), then updating the following (**make sure to use fully-qualified (i.e. absolute) paths**):

1. Path to PID file (`<path to pidfile>`)

    The path to the PID file should be the same one that was specified in the configuration file.

2. Path to Python executable (`<path to python executable>`)

    Locate the Python executable you would like to use. This will help fix the version of Python that is used, along with its environments.
    
    If you're using a virtual environment, make sure to activate the environment first. If you're not, then proceed directly to the following.

    You can use either the shell's `which` command:

        which python

    or find the path of the executable using Python itself:

        python -c "import sys; print(sys.executable);"


3. Path to `segmenter` repository (`<path to segmenter repo>`)

    This refers to this very repository (I don't know how to make it any clearer). And in case it's ambiguous for whatever reasons, only the path to the repository's root directory is required, not any subdirectories.

4. Path to JSON configuration file (`<path to JSON configuration file>`)

    This refers to the configuration file that you created in the step immediately before this. **Not the original configuration file that you made copied from, but the copy that you made and then updated with the appropriate values.** (If you didn't make a copy of the configuration file and chose to edit it directly, then I am at a lost for words.)

There could be additional optional modifications you can choose to make:

1. Editing the description of the service
2. Changing the `--server` option upon microservice start in the `ExecStart` command.

    This will change the forking strategy that is used when serving requests, i.e. whether to use processes or threads. If you know what you're doing then great, otherwise, it's your own funeral.

3. Supporting automatic restarting with the `Restart` option in the service file.

Once you've configured the service file, move/symlink it to `/etc/systemd/system` (or the corresponding location on non-RHEL systems). Be careful what you name the file, as `systemd` will derive the service name from the filename (e.g. `/etc/systemd/system/hello.service` will have service name `hello`).

Once that is done, let `systemd` know of the newly added service file by running:

    systemctl daemon-reload

Assuming nothing went wrong, you should now be able to use `systemd` to manage the microservice for you.

To start the service:

    systemctl start <service name>

To stop the service:

    systemctl stop <service name>

To check the status of the service:

    systemctl status <service name>

And lastly, to start the service on boot:

    systemctl enable <service name>

