Datasets
========

NeMo has scripts to convert several common ASR datasets into the format expected by the ``nemo_asr`` collection. You can get started
with those datasets by following the instructions to run those scripts in the section appropriate to each dataset below.

If the user has their own data and want to preprocess it to use with NeMo ASR models, refer to the `Preparing Custom ASR Data`_ section.

If the user already has a dataset that you want to convert to a tarred format, refer to the :ref:`Tarred Datasets <Tarred_Datasets>` section.

.. _LibriSpeech_dataset:

LibriSpeech
-----------

Run the following scripts to download the LibriSpeech data and convert it into the format expected by `nemo_asr`. At least 250GB free
space is required.

.. code-block:: bash

    # install sox
    sudo apt-get install sox
    mkdir data
    python get_librispeech_data.py --data_root=data --data_set=ALL

After this, the ``data`` folder should contain wav files and ``.json`` manifests for NeMo ASR datalayer.

Each line is a training example. ``audio_filepath`` contains the path to the wav file, ``duration`` is the duration in seconds, and ``text`` is the transcript:

.. code-block:: json

    {"audio_filepath": "<absolute_path_to>/1355-39947-0000.wav", "duration": 11.3, "text": "psychotherapy and the community both the physician and the patient find their place in the community the life interests of which are superior to the interests of the individual"}
    {"audio_filepath": "<absolute_path_to>/1355-39947-0001.wav", "duration": 15.905, "text": "it is an unavoidable question how far from the higher point of view of the social mind the psychotherapeutic efforts should be encouraged or suppressed are there any conditions which suggest suspicion of or direct opposition to such curative work"}

Fisher English Training Speech
------------------------------

Run these scripts to convert the Fisher English Training Speech data into a format expected by the ``nemo_asr`` collection.

In brief, the following scripts convert the ``.sph`` files to ``.wav``, slices those files into smaller audio samples, matches the
smaller slices with their corresponding transcripts, and splits the resulting audio segments into train, validation, and test sets
(with one manifest each).

.. note::
  - 106 GB of space is required to run the ``.wav`` conversion
  - additional 105 GB is required for the slicing and matching
  - ``sph2pipe`` is required in order to run the ``.wav`` conversion

**Instructions**

The following scripts assume that you already have the Fisher dataset from the Linguistic Data Consortium, with a directory structure
that looks similar to the following:

.. code-block:: bash

  FisherEnglishTrainingSpeech/
  ├── LDC2004S13-Part1
  │   ├── fe_03_p1_transcripts
  │   ├── fisher_eng_tr_sp_d1
  │   ├── fisher_eng_tr_sp_d2
  │   ├── fisher_eng_tr_sp_d3
  │   └── ...
  └── LDC2005S13-Part2
      ├── fe_03_p2_transcripts
      ├── fe_03_p2_sph1
      ├── fe_03_p2_sph2
      ├── fe_03_p2_sph3
      └── ...

The transcripts that will be used are located in the ``fe_03_p<1,2>_transcripts/data/trans`` directory. The audio files (``.sph``)
are located in the remaining directories in an ``audio`` subdirectory.

#. Convert the audio files from ``.sph`` to ``.wav`` by running:

   .. code-block:: bash

     cd <nemo_root>/scripts/dataset_processing
     python fisher_audio_to_wav.py \
       --data_root=<fisher_root> --dest_root=<conversion_target_dir>

   This will place the unsliced ``.wav`` files in ``<conversion_target_dir>/LDC200[4,5]S13-Part[1,2]/audio-wav/``. It will take several
   minutes to run.

#. Process the transcripts and slice the audio data.

   .. code-block:: bash

     python process_fisher_data.py \
       --audio_root=<conversion_target_dir> --transcript_root=<fisher_root> \
       --dest_root=<processing_target_dir> \
       --remove_noises

   This script splits the full dataset into train, validation, test sets, and places the audio slices in the corresponding folders
   in the destination directory. One manifest is written out per set, which includes each slice's transcript, duration, and path.

   This will likely take around 20 minutes to run. Once finished, delete the 10 minute long ``.wav`` files.

2000 HUB5 English Evaluation Speech
-----------------------------------

Run the following script to convert the HUB5 data into a format expected by the ``nemo_asr`` collection.

Similarly, to the Fisher dataset processing scripts, this script converts the ``.sph`` files to ``.wav``, slices the audio files and
transcripts into utterances, and combines them into segments of some minimum length (default is 10 seconds). The resulting segments
are all written out to an audio directory and the corresponding transcripts are written to a manifest JSON file.

.. note::
  - 5 GB of free space is required to run this script
  - ``sph2pipe`` is also required to be installed

This script assumes you already have the 2000 HUB5 dataset from the Linguistic Data Consortium.

Run the following command to process the 2000 HUB5 English Evaluation Speech samples:

.. code-block:: bash

  python process_hub5_data.py \
    --data_root=<path_to_HUB5_data> \
    --dest_root=<target_dir>

You can optionally include ``--min_slice_duration=<num_seconds>`` if you would like to change the minimum audio segment duration.

AN4 Dataset
-----------

This is a small dataset recorded and distributed by Carnegie Mellon University. It consists of recordings of people spelling out
addresses, names, etc.

#. `Download and extract the dataset <https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz>`_ (which is labeled "NIST's Sphere audio (.sph) format (64M)".

#. Convert the ``.sph`` files to ``.wav`` using sox, and build one training and one test manifest.

   .. code-block:: bash

     python process_an4_data.py --data_root=<path_to_extracted_data>

After the script finishes, the ``train_manifest.json`` and ``test_manifest.json`` can be found in the ``<data_root>/an4/`` directory.

Aishell-1
---------

To download the Aishell-1 data and convert it into a format expected by ``nemo_asr``, run:

.. code-block:: bash

    # install sox
    sudo apt-get install sox
    mkdir data
    python get_aishell_data.py --data_root=data

After the script finishes, the ``data`` folder should contain a ``data_aishell`` folder which contains a wav file, a transcript folder,  and related ``.json`` and ``vocab.txt`` files.

Aishell-2
---------

To process the AIShell-2 dataset, in the command below, set the data folder of AIShell-2 using ``--audio_folder`` and where to push
these files using ``--dest_folder``. In order to generate files in the supported format of ``nemo_asr``, run:

.. code-block:: bash

    python process_aishell2_data.py --audio_folder=<data directory> --dest_folder=<destination directory>

After the script finishes, the ``train.json``, ``dev.json``, ``test.json``, and ``vocab.txt`` files can be found in the ``dest_folder`` directory.

.. _section-with-manifest-format-explanation:

Preparing Custom ASR Data
-------------------------

The ``nemo_asr`` collection expects each dataset to consist of a set of utterances in individual audio files plus
a manifest that describes the dataset, with information about one utterance per line (``.json``).
The audio files can be of any format supported by `Pydub <https://github.com/jiaaro/pydub>`_, though we recommend
WAV files as they are the default and have been most thoroughly tested.

There should be one manifest file per dataset that will be passed in, therefore, if the user wants separate training and validation
datasets, they should also have separate manifests. Otherwise, they will be loading validation data with their training data and vice
versa.

Each line of the manifest should be in the following format:

.. code-block:: json

  {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}

The :code:`audio_filepath` field should provide an absolute path to the ``.wav`` file corresponding to the utterance.
The :code:`text` field should contain the full transcript for the utterance, and the :code:`duration` field should
reflect the duration of the utterance in seconds.

Each entry in the manifest (describing one audio file) should be bordered by '{' and '}' and must
be contained on one line. The fields that describe the file should be separated by commas, and have the form :code:`"field_name": value`,
as shown above. There should be no extra lines in the manifest, i.e. there should be exactly as many lines in the manifest as
there are audio files in the dataset.

Since the manifest specifies the path for each utterance, the audio files do not have to be located
in the same directory as the manifest, or even in any specific directory structure.

Once there is a manifest that describes each audio file in the dataset, use the dataset by passing
in the manifest file path in the experiment config file, e.g. as ``training_ds.manifest_filepath=<path/to/manifest.json>``.

.. _Tarred_Datasets:

Tarred Datasets
---------------

If experiments are run on a cluster with datasets stored on a distributed file system, the user will likely
want to avoid constantly reading multiple small files and would prefer tarring their audio files.
There are tarred versions of some NeMo ASR dataset classes for this case, such as the ``TarredAudioToCharDataset``
(corresponding to the ``AudioToCharDataset``) and the ``TarredAudioToBPEDataset`` (corresponding to the
``AudioToBPEDataset``). The tarred audio dataset classes in NeMo use `WebDataset <https://github.com/tmbdev/webdataset>`_.

To use an existing tarred dataset instead of a non-tarred dataset, set ``is_tarred: true`` in
the experiment config file. Then, pass in the paths to all of the audio tarballs in ``tarred_audio_filepaths``, either as a list
of filepaths, e.g. ``['/data/shard1.tar', '/data/shard2.tar']``, or in a single brace-expandable string, e.g.
``'/data/shard_{1..64}.tar'`` or ``'/data/shard__OP_1..64_CL_'`` (recommended, see note below).

.. note::
  For brace expansion, there may be cases where ``{x..y}`` syntax cannot be used due to shell interference. This occurs most commonly
  inside SLURM scripts. Therefore, we provide a few equivalent replacements. Supported opening braces (equivalent to ``{``) are ``(``,
  ``[``, ``<`` and the special tag ``_OP_``. Supported closing braces (equivalent to ``}``) are ``)``, ``]``, ``>`` and the special
  tag ``_CL_``. For SLURM based tasks, we suggest the use of the special tags for ease of use.

As with non-tarred datasets, the manifest file should be passed in ``manifest_filepath``. The dataloader assumes that the length
of the manifest after filtering is the correct size of the dataset for reporting training progress.

The ``tarred_shard_strategy`` field of the config file can be set if you have multiple shards and are running an experiment with
multiple workers. It defaults to ``scatter``, which preallocates a set of shards per worker which do not change during runtime.
Note that this strategy, on specific occasions (when the number of shards is not divisible with ``world_size``), will not sample
the entire dataset. As an alternative the ``replicate`` strategy, will preallocate the entire set of shards to every worker and not
change it during runtime. The benefit of this strategy is that it allows each worker to sample data points from the entire dataset
independently of others. Note, though, that more than one worker may sample the same shard, and even sample the same data points!
As such, there is no assured guarantee that all samples in the dataset will be sampled at least once during 1 epoch. Note that
for these reasons it is not advisable to use tarred datasets as validation and test datasets.

For more information about the individual tarred datasets and the parameters available, including shuffling options,
see the corresponding class APIs in the :ref:`Datasets <asr-api-datasets>` section.

.. warning::
  If using multiple workers, the number of shards should be divisible by the world size to ensure an even
  split among workers. If it is not divisible, logging will give a warning but training will proceed, but likely hang at the last epoch.
  In addition, if using distributed processing, each shard must have the same number of entries after filtering is
  applied such that each worker ends up with the same number of files. We currently do not check for this in any dataloader, but the user's
  program may hang if the shards are uneven.

Sharded Manifests
~~~~~~~~~~~~~~~~~
If your dataset / manifest is large, you may wish to use sharded manifest files instead of a single manifest file. The naming convention
is identical to the audio tarballs and there should be a 1:1 relationship between a sharded audio tarfile and its manifest shard; e.g.
``'/data/sharded_manifests/manifest__OP_1..64_CL_'`` in the above example. Using sharded manifests improves job startup times and
decreases memory usage, as each worker only loads manifest shards for the corresponding audio shards instead of the entire manifest.

To enable sharded manifest filename expansion, set the ``shard_manifests`` field of the config file to true. In addition, the
``defer_setup`` flag needs to be true as well, so that the dataloader will be initialized after the DDP and its length can be collected from
the distributed workers.

Batching strategies
---------------------

For training ASR models, audios with different lengths may be grouped into a batch. It would make it necessary to use paddings to make all the same length.
These extra paddings is a significant source of computation waste. 

Semi Sorted Batching
---------------------

Sorting samples by duration and spliting them into batches speeds up training, but can degrade the quality of the model. To avoid quality degradation and maintain some randomness in the partitioning process, we add pseudo noise to the sample length when sorting.

It may result into training speeedup of more than 40 percent with the same quality. To enable and use semi sorted batching add some lines in config.

  .. code::

    ++model.train_ds.use_semi_sorted_batching=true
    ++model.train_ds.randomization_factor=0.1

Semi sorted batching is supported by the following models:

  .. code::

    nemo.collections.asr.models.EncDecCTCModel
    nemo.collections.asr.models.EncDecCTCModelBPE
    nemo.collections.asr.models.EncDecRNNTModel
    nemo.collections.asr.models.EncDecRNNTBPEModel
    nemo.collections.asr.models.EncDecHybridRNNTCTCModel
    nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModel

For more details about this algorithm, see the `paper <https://www.isca-archive.org/interspeech_2021/ge21_interspeech.pdf>`_ .

.. _Bucketing_Datasets:

Bucketing Datasets
---------------------

Splitting the training samples into buckets with different lengths and sampling from the same bucket for each batch would increase the computation efficicncy.
It may result into training speeedup of more than 2X. To enable and use the bucketing feature, you need to create the bucketing version of the dataset by using `conversion script here <https://github.com/NVIDIA/NeMo/tree/stable/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_.
You may use --buckets_num to specify the number of buckets (Recommend to use 4 to 8 buckets). It creates multiple tarred datasets, one per bucket, based on the audio durations. The range of [min_duration, max_duration) is split into equal sized buckets.

To enable the bucketing feature in the dataset section of the config files, you need to pass the multiple tarred datasets as a list of lists.
If user passes just a list of strings, then the datasets would simply get concatenated which would be different from bucketing.
Here is an example for 4 buckets and 512 shards:

.. code::

    python speech_to_text_bpe.py
    ...
    model.train_ds.manifest_filepath=[[PATH_TO_TARS/bucket1/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket2/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket3/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket4/tarred_audio_manifest.json]]
    model.train_ds.tarred_audio_filepaths=[[PATH_TO_TARS/bucket1/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket2/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket3/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket4/audio__OP_0..511_CL_.tar]]

When bucketing is enabled, in each epoch, first all GPUs would use the first bucket, then go to the second bucket, and so on. It guarantees that all GPUs are using the same bucket at the same time. It reduces the number of paddings in each batch and speedup the training significantly without hurting the accuracy significantly.

There are two types of batching:

*  Fixed-size bucketing: all batches would have the same number of samples specified by train_ds.batch_size
*  Adaptive-size bucketing: uses different batch sizes for each bucket.

Adaptive-size bucketing helps to increase the GPU utilization and speedup the training.
Batches sampled from buckets with smaller audio lengths can be larger which would increase the GPU utilization and speedup the training.
You may use train_ds.bucketing_batch_size to enable the adaptive batching and specify the batch sizes for the buckets.
When bucketing_batch_size is not set, train_ds.batch_size is going to be used for all buckets (fixed-size bucketing).

bucketing_batch_size can be set as an integer or a list of integers to explicitly specify the batch size for each bucket.
if bucketing_batch_size is set to be an integer, then linear scaling is being used to scale-up the batch sizes for batches with shorted audio size. For example, setting train_ds.bucketing_batch_size=8 for 4 buckets would use these sizes [32,24,16,8] for different buckets.
When bucketing_batch_size is set, traind_ds.batch_size need to be set to 1.

Training an ASR model on audios sorted based on length may affect the accuracy of the model. We introduced some strategies to mitigate it.
We support three types of bucketing strategies:

*   fixed_order: the same order of buckets are used for all epochs
*   synced_randomized (default): each epoch would have a different order of buckets. Order of the buckets is shuffled every epoch.
*   fully_randomized: similar to synced_randomized but each GPU has its own random order. So GPUs would not be synced.

Tha parameter train_ds.bucketing_strategy can be set to specify one of these strategies. The recommended strategy is synced_randomized which gives the highest training speedup.
The fully_randomized strategy would have lower speedup than synced_randomized but may give better accuracy.

Bucketing may improve the training speed more than 2x but may affect the final accuracy of the model slightly. Training for more epochs and using 'synced_randomized' strategy help to fill this gap.
Currently bucketing feature is just supported for tarred datasets.


Conversion to Tarred Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can easily convert your existing NeMo-compatible ASR datasets using the
`conversion script here <https://github.com/NVIDIA/NeMo/tree/stable/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_.

.. code:: bash

  python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the manifest file> \
    --target_dir=<path to output directory> \
    --num_shards=<number of tarfiles that will contain the audio>
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=0

This script shuffles the entries in the given manifest (if ``--shuffle`` is set, which we recommend), filter
audio files according to ``min_duration`` and ``max_duration``, and tar the remaining audio files to the directory
``--target_dir`` in ``n`` shards, along with separate manifest and metadata files.

The files in the target directory should look similar to the following:

.. code::

  target_dir/
  ├── audio_1.tar
  ├── audio_2.tar
  ├── ...
  ├── metadata.yaml
  ├── tarred_audio_manifest.json
  ├── sharded_manifests/
      ├── manifest_1.json
      ├── ...
      └── manifest_N.json


Note that file structures are flattened such that all audio files are at the top level in each tarball. This ensures that
filenames are unique in the tarred dataset and the filepaths do not contain "-sub" and forward slashes in each ``audio_filepath`` are
simply converted to underscores. For example, a manifest entry for ``/data/directory1/file.wav`` would be ``_data_directory1_file.wav``
in the tarred dataset manifest, and ``/data/directory2/file.wav`` would be converted to ``_data_directory2_file.wav``.

Sharded manifests are generated by default; this behavior can be toggled via the ``no_shard_manifests`` flag.

Upsampling Datasets
-------------------

Buckets may also be 'weighted' to allow multiple runs through a target dataset during each training epoch. This can be beneficial in cases when a dataset is composed of several component sets of unequal sizes and one desires to mitigate bias towards the larger sets through oversampling.

Weighting is managed with the `bucketing_weights` parameter. After passing your composite tarred datasets in the format described above for bucketing, pass a list of integers (one per bucket) to indicate how many times a manifest should be read during training.

For example, by passing `[2,1,1,3]` to the code below:

.. code::

    python speech_to_text_bpe.py
    ...
    model.train_ds.manifest_filepath=[[PATH_TO_TARS/bucket1/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket2/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket3/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket4/tarred_audio_manifest.json]]
    model.train_ds.tarred_audio_filepaths=[[PATH_TO_TARS/bucket1/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket2/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket3/audio__OP_0..511_CL_.tar],
    [PATH_TO_TARS/bucket4/audio__OP_0..511_CL_.tar]]
	...
	model.train_ds.bucketing_weights=[2,1,1,3]

NeMo will configure training so that all data in `bucket1` will be present twice in a training epoch, `bucket4` will be present three times, and that of `bucket2` and `bucket3` will occur only once each. Note that this will increase the effective amount of data present during training and thus affect training time per epoch.

If using adaptive bucketing, note that the same batch size will be assigned to each instance of the upsampled data. That is, given the following:

.. code::

    python speech_to_text_bpe.py
    ...
    model.train_ds.manifest_filepath=[[PATH_TO_TARS/bucket1/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket2/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket3/tarred_audio_manifest.json],
    [PATH_TO_TARS/bucket4/tarred_audio_manifest.json]]
	...
	...
	model.train_ds.bucketing_weights=[2,1,1,3]
	model.train_ds.bucketing_batch_size=[4,4,4,2]

All instances of data from `bucket4` will still be trained with a batch size of 2 while all others would have a batch size of 4. As with standard bucketing, this requires `batch_size`` to be set to 1.
If `bucketing_batch_size` is not specified, all datasets will be passed with the same fixed batch size as specified by the `batch_size` parameter.

It is recommended to set bucketing strategies to `fully_randomized` during multi-GPU training to prevent possible dataset bias during training.


Datasets on AIStore
-------------------

`AIStore <https://aiatscale.org>`_ is an open-source lightweight object storage system focused on large-scale deep learning.
AIStore is aimed to scale linearly with each added storage node, can be deployed on any Linux machine and can provide a unified namespace across multiple remote backends, such as Amazon S3, Google Cloud, and Microsoft Azure.
More details are provided in the `documentation <https://aiatscale.org/docs>`_ and the `repository <https://github.com/NVIDIA/aistore>`_ of the AIStore project.

NeMo currently supports datasets from an AIStore bucket provider under ``ais://`` namespace.

AIStore Setup
~~~~~~~~~~~~~

NeMo is currently relying on the AIStore (AIS) command-line interface (CLI) to handle the supported datasets.
The CLI is available in current NeMo Docker containers.
If necessary, the CLI can be configured using the instructions provided in `AIStore CLI <https://aiatscale.org/docs/cli>`_ documentation.

To start using the AIS CLI to access data on an AIS cluster, an endpoint needs to be configured.
The endpoint is configured by setting ``AIS_ENDPOINT`` environment variable before using the CLI

.. code::

    export AIS_ENDPOINT=http://hostname:port
    ais --help

In the above, ``hostname:port`` denotes the address of an AIS gateway.
For example, the address could be ``localhost:51080`` if testing using a local `minimal production-ready standalone Docker container <https://github.com/NVIDIA/aistore/blob/master/deploy/prod/docker/single/README.md>`_.

Dataset Setup
~~~~~~~~~~~~~

Currently, both tarred and non-tarred datasets are supported.
For any dataset, the corresponding manifest file is cached locally and processed as a regular manifest file.
For non-tarred datasets, the audio data is also cached locally.
For tarred datasets, shards from the AIS cluster are used by piping ``ais get`` to WebDataset.

Tarred Dataset from AIS
^^^^^^^^^^^^^^^^^^^^^^^

A tarred dataset can be easily used as described in the :ref:`Tarred Datasets <Tarred_Datasets>` section by providing paths to manifests on an AIS cluster.
For example, a tarred dataset from an AIS cluster can be configured as

.. code::

  manifest_filepath='ais://bucket/tarred_audio_manifest.json'
  tarred_audio_filepaths='ais://bucket/shard_{1..64}.tar'

:ref:`Bucketing Datasets <Bucketing_Datasets>` are configured in a similar way by providing paths on an AIS cluster.

Non-tarred Dataset from AIS
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A non-tarred dataset can be easly used by providing a manifest file path on an AIS cluster

.. code::

  manifest_filepath='ais://bucket/dataset_manifest.json'

Note that it is assumed that the manifest file path contains audio file paths relative to the manifest locations.
For example the manifest file may have lines in the following format

.. code-block:: json

  {"audio_filepath": "path/to/audio.wav", "text": "transcription of the uterance", "duration": 23.147}

The corresponding audio file would be downloaded from ``ais://bucket/path/to/audio.wav``.

Cache configuration
^^^^^^^^^^^^^^^^^^^

Manifests and audio files from non-tarred datasets will be cached locally.
Location of the cache can be configured by setting two environment variables

- ``NEMO_DATA_STORE_CACHE_DIR``: path to a location which can be used to cache the data
- ``NEMO_DATA_STORE_CACHE_SHARED``: flag to denote whether the cache location is shared between the compute nodes

In a multi-node environment, the cache location may or may be not shared between the nodes.
This can be configured by setting ``NEMO_DATA_STORE_CACHE_SHARED`` to ``1`` when the location is shared between the nodes or to ``0`` when each node has a separate cache.

When a globally shared cache is available, the data should be cached only once from the global rank zero node.
When a node-specific cache is used, the data should be cached only once by each local rank zero node.
To control this behavior using `torch.distributed.barrier`, instantiation of the corresponding dataloader needs to be deferred ``ModelPT::setup``, to ensure a distributed environment has been initialized.
This can be achieved by setting ``defer_setup`` as

.. code:: shell

  ++model.train_ds.defer_setup=true
  ++model.validation_ds.defer_setup=true
  ++model.test_ds.defer_setup=true


Complete Example
^^^^^^^^^^^^^^^^

An example using an AIS cluster at ``hostname:port`` with a tarred dataset for training, a non-tarred dataset for validation and node-specific caching is given below

.. code:: shell

  export AIS_ENDPOINT=http://hostname:port \
  && export NEMO_DATA_STORE_CACHE_DIR=/tmp \
  && export NEMO_DATA_STORE_CACHE_SHARED=0 \
  python speech_to_text_bpe.py \
  ...
  model.train_ds.manifest_filepath=ais://train_bucket/tarred_audio_manifest.json \
  model.train_ds.tarred_audio_filepaths=ais://train_bucket/audio__OP_0..511_CL_.tar \
  ++model.train_ds.defer_setup=true \
  mode.validation_ds.manifest_filepath=ais://validation_bucket/validation_manifest.json \
  ++model.validation_ds.defer_setup=true


.. _Hybrid-ASR-TTS_model__Text-Only-Data:


Lhotse Dataloading
------------------

NeMo supports using `Lhotse`_, a speech data handling library, as a dataloading option. The key features of Lhotse used in NeMo are:

* Dynamic batch sizes
    Lhotse samples mini-batches to satisfy the constraint of total speech duration in a mini-batch (``batch_duration``),
    rather than a specific number of examples (i.e., batch size).
* Dynamic bucketing
    Instead of statically pre-bucketing the data, Lhotse allocates training examples to buckets dynamically.
    This allows more rapid experimentation with bucketing settings (number of buckets, specific placement of bucket duration bins)
    to minimize the amount of padding and accelerate training.
* Quadratic duration penalty
    Adding a quadratic penalty to an utterance's duration allows to sample mini-batches so that the
    GPU utilization is more consistent across big batches of short utterances and small batches of long utterances when using
    models with quadratic time/memory complexity (such as transformer).
* Dynamic weighted data source multiplexing
    An approach to combining diverse data sources (e.g. multiple domains, languages, tasks)
    where each data source is treated as a separate stream with its own sampling probability. The resulting data stream is a
    multiplexer that samples from each sub-stream. This approach ensures that the distribution of different sources is approximately
    constant in time (i.e., stationary); in fact, each mini-batch will have roughly the same ratio of data coming from each source.
    Since the multiplexing is done dynamically, it is very easy to tune the sampling weights.

Lhotse dataloading supports the following types of inputs:

* NeMo manifests
    Regular NeMo JSON manifests.
* NeMo tarred data
    Tarred NeMo JSON manifests + audio tar files; we also support combination of multiple NeMo
    tarred data sources (e.g., multiple buckets of NeMo data or multiple datasets) via dynamic multiplexing.

    We support using a subset of Tarred NeMo JSON manifests along with audio tar files without disrupting the alignment between the tarred files and their corresponding manifests.
    This feature is essential because large datasets often consist of numerous tar files and multiple versions of Tarred NeMo JSON manifest subsets, which may contain only a portion of the audio files due to filtering for various reasons.
    To skip specific entries in the manifests without repeatedly copying and retarring audio files, the entries must include a ``_skipme`` key. This key should be set to ``True``, ``1``, or a reason for skipping (e.g., ``low character-rate``).

* Lhotse CutSet manifests
    Regular Lhotse CutSet manifests (typically gzipped JSONL).
    See `Lhotse Cuts documentation`_ to learn more about Lhotse data formats.
* Lhotse Shar data
    Lhotse Shar is a data format that also uses tar files for sequential data loading,
    but is designed to be modular (i.e., easily extensible with new data sources and with new feature fields).
    More details can be found here: |tutorial_shar|

.. caution:: As of now, Lhotse is mainly supported in most ASR model configurations. We aim to gradually extend this support to other speech tasks.

.. _Lhotse: https://github.com/lhotse-speech/lhotse
.. _Lhotse Cuts documentation: https://lhotse.readthedocs.io/en/latest/cuts.html
.. |tutorial_shar| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb

Enabling Lhotse via configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: Using Lhotse with tarred datasets will make the dataloader infinite, ditching the notion of an "epoch". "Epoch" may still be logged in W&B/TensorBoard, but it will correspond to the number of executed training loops between validation loops.

Start with an existing NeMo experiment YAML configuration. Typically, you'll only need to add a few options to enable Lhotse.
These options are::

    # NeMo generic dataloading arguments
    model.train_ds.manifest_filepath=...
    model.train_ds.tarred_audio_filepaths=...   # for tarred datasets only
    model.train_ds.num_workers=4
    model.train_ds.min_duration=0.3             # optional
    model.train_ds.max_duration=30.0            # optional
    model.train_ds.shuffle=true                 # optional

    # Lhotse dataloading related arguments
    ++model.train_ds.use_lhotse=True
    ++model.train_ds.batch_duration=1100
    ++model.train_ds.quadratic_duration=30
    ++model.train_ds.num_buckets=30
    ++model.train_ds.num_cuts_for_bins_estimate=10000
    ++model.train_ds.bucket_buffer_size=10000
    ++model.train_ds.shuffle_buffer_size=10000

    # PyTorch Lightning related arguments
    ++trainer.use_distributed_sampler=false
    ++trainer.limit_train_batches=1000
    trainer.val_check_interval=1000
    trainer.max_steps=300000

.. note:: The default values above are a reasonable starting point for a hybrid RNN-T + CTC ASR model on a 32GB GPU with a data distribution dominated by 15s long utterances.

Let's briefly go over each of the Lhotse dataloading arguments:

* ``use_lhotse`` enables Lhotse dataloading
* ``batch_duration`` is the total max duration of utterances in a mini-batch and controls the batch size; the more shorter utterances, the bigger the batch size, and vice versa.
* ``quadratic_duration`` adds a quadratically growing penalty for long utterances; useful in bucketing and transformer type of models. The value set here means utterances this long will count as if with a doubled duration.
* ``num_buckets`` is the number of buckets in the bucketing sampler. Bigger value means less padding but also less randomization.
* ``num_cuts_for_bins_estimate`` is the number of utterance we will sample before the start of the training to estimate the duration bins for buckets. Larger number results in a more accurate estimatation but also a bigger lag before starting the training.
* ``bucket_buffer_size`` is the number of utterances (data and metadata) we will hold in memory to be distributed between buckets. With bigger ``batch_duration``, this number may need to be increased for dynamic bucketing sampler to work properly (typically it will emit a warning if this is too low).
* ``shuffle_buffer_size`` is an extra number of utterances we will hold in memory to perform approximate shuffling (via reservoir-like sampling). Bigger number means more memory usage but also better randomness.

The PyTorch Lightning ``trainer`` related arguments:

* ``use_distributed_sampler=false`` is required because Lhotse has its own handling of distributed sampling.
* ``val_check_interval``/``limit_train_batches``
    These are required for dataloaders with tarred/Shar datasets
    because Lhotse makes the dataloader infinite, so we'd never go past epoch 0. This approach guarantees
    we will never hang the training because the dataloader in some node has less mini-batches than the others
    in some epochs. The value provided here will be the effective length of each "pseudo-epoch" after which we'll
    trigger the validation loop.
* ``max_steps`` is the total number of steps we expect to be training for. It is required for the same reason as ``limit_train_batches``; since we'd never go past epoch 0, the training would have never finished.

Some other Lhotse related arguments we support:

* ``cuts_path`` can be provided to read data from a Lhotse CutSet manifest instead of a NeMo manifest.
    Specifying this option will result in ``manifest_filepaths`` and ``tarred_audio_filepaths`` being ignored.
* ``shar_path``
    Can be provided to read data from a Lhotse Shar manifest instead of a NeMo manifest.
    Specifying this option will result in ``manifest_filepaths`` and ``tarred_audio_filepaths`` being ignored.
    This argument can be a string (single Shar directory), a list of strings (Shar directories),
    or a list of 2-item lists, where the first item is a Shar directory path, and the other is a sampling weight.
    The user can also provide a dict mapping Lhotse Shar fields to a list of shard paths with data for that field.
    For details about Lhotse Shar format, see: |tutorial_shar|
* ``bucket_duration_bins``
    Duration bins are a list of float values (seconds) that when provided, will skip the initial bucket bin estimation
    and save some time. It has to have a length of ``num_buckets - 1``. An optimal value can be obtained by running CLI:
    ``lhotse cut estimate-bucket-bins -b $num_buckets my-cuts.jsonl.gz``
* ``use_bucketing`` is a boolean which indicates if we want to enable/disable dynamic bucketing. By defalt it's enabled.
* ``text_field`` is the name of the key in the JSON (NeMo) manifest from which we should be reading text (default="text").
* ``lang_field`` is the name of the key in the JSON (NeMo) manifest from which we should be reading language tag (default="lang"). This is useful when working e.g. with ``AggregateTokenizer``.
* ``batch_size``
    Limits the number of examples in a mini-batch to this number, when combined with ``batch_duration``.
    When ``batch_duration`` is not set, it acts as a static batch size.
* ``seed`` sets a random seed for the shuffle buffer.

The full and always up-to-date list of supported options can be found in ``LhotseDataLoadingConfig`` class.

Extended multi-dataset configuration format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combining a large number of datasets and defining weights for them can be tricky.
We offer an extended configuration format that allows you to explicitly define datasets,
dataset groups, and their weights either inline in the experiment configuration,
or as a path to a separate YAML file.

In addition to the features above, this format introduces a special ``tags`` dict-like field.
The keys and values in ``tags`` are automatically attached to every sampled example, which
is very useful when combining multiple datasets with different properties.
The dataset class which converts these examples to tensors can partition the mini-batch and apply
different processing to each group.
For example, you may want to construct different prompts for the model using metadata in ``tags``.

.. note:: When fine-tuning a model that was trained with ``input_cfg`` option, typically you'd only need
    to override the following options: ``input_cfg=null`` and ``manifest_filepath=path/to/manifest.json``.

Example 1. Combine two datasets with equal weights and attach custom metadata in ``tags`` to each cut:

.. code-block:: yaml

    input_cfg:
      - type: nemo_tarred
        manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
        tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
        weight: 0.4
        tags:
          lang: en
          pnc: no
      - type: nemo_tarred
        manifest_filepath: /path/to/other/manifest__OP_0..512_CL_.json
        tarred_audio_filepath: /path/to/other/tarred_audio/audio__OP_0..512_CL_.tar
        weight: 0.6
        tags:
          lang: pl
          pnc: yes

Example 2. Combine multiple (4) datasets, corresponding to different tasks (ASR, AST).
Each task gets its own group and its own weight.
Then within each task, each dataset get its own within-group weight as well.
The final weight is the product of outer and inner weight:

.. code-block:: yaml

    input_cfg:
      - type: group
        weight: 0.7
        tags:
          task: asr
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/asr1/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/tarred_audio/asr1/audio__OP_0..512_CL_.tar
            weight: 0.6
            tags:
              source_lang: en
              target_lang: en
          - type: nemo_tarred
            manifest_filepath: /path/to/asr2/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/asr2/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.4
            tags:
              source_lang: pl
              target_lang: pl
      - type: group
        weight: 0.3
        tags:
          task: ast
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/ast1/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/ast1/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.2
            tags:
              source_lang: en
              target_lang: pl
          - type: nemo_tarred
            manifest_filepath: /path/to/ast2/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/ast2/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.8
            tags:
              source_lang: pl
              target_lang: en

Configuring multimodal dataloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our configuration format supports specifying data sources from other modalities than just audio.
At this time, this support is extended to audio and text modalities. We provide the following parser types:

**Raw text files.** Simple text files where each line is an individual text example. This can represent standard language modeling data.
This parser is registered under ``type: txt``.

Data format examples::

    # file: document_0.txt
    This is a language modeling example.
    Wall Street is expecting major news tomorrow.

    # file: document_1.txt
    Invisible bats have stormed the city.
    What an incredible event!

Dataloading configuration example::

    input_cfg:
      - type: txt
        paths: /path/to/document_{0..1}.txt
        language: en  # optional

Python object example::

    from nemo.collections.common.data.lhotse.text_adapters import TextExample

    example = TextExample(
        text="This is a language modeling example.",
        language="en",  # optional
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {"type": "txt", "paths": "/path/to/document_{0..1}.txt", "language": "en"},
            ],
            "use_multimodal_dataloading": True,
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**Raw text file pairs.** Pairs of raw text files with corresponding lines. This can represent machine translation data.
This parser is registered under ``type: txt_pair``.

Data format examples::

    # file: document_en_0.txt
    This is a machine translation example.
    Wall Street is expecting major news tomorrow.

    # file: document_pl_0.txt
    To jest przykład tłumaczenia maszynowego.
    Wall Street spodziewa się jutro ważnych wiadomości.

Dataloading configuration example::

    input_cfg:
      - type: txt_pair
        source_path: /path/to/document_en_{0..N}.txt
        target_path: /path/to/document_pl_{0..N}.txt
        source_language: en  # optional
        target_language: pl  # optional

Python object example::

    from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample

    example = SourceTargetTextExample(
        source=TextExample(
            text="This is a language modeling example.",
            language="en",  # optional
        ),
        target=TextExample(
            text="To jest przykład tłumaczenia maszynowego.",
            language="pl",  # optional
        ),
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_path": "/path/to/document_en_{0..N}.txt",
                    "target_path": "/path/to/document_pl_{0..N}.txt",
                    "source_language": "en"
                    "target_language": "en"
                },
            ],
            "use_multimodal_dataloading": True,
            "prompt_format": "t5nmt",
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**NeMo multimodal conversations.** A JSON-Lines (JSONL) file that defines multi-turn conversations with mixed text and audio turns.
This parser is registered under ``type: multimodal_conversation``.

Data format examples::

    # file: chat_0.jsonl
    {"id": "conv-0", "conversations": [{"from": "user", "value": "speak to me", "type": "text"}, {"from": "assistant": "value": "/path/to/audio.wav", "duration": 17.1, "type": "audio"}]}

Dataloading configuration example::

    token_equivalent_duration: 0.08
    input_cfg:
      - type: multimodal_conversation
        manifest_filepath: /path/to/chat_{0..N}.jsonl
        audio_locator_tag: [audio]

Python object example::

    from lhotse import Recording
    from nemo.collections.common.data.lhotse.text_adapters import MultimodalConversation, TextTurn, AudioTurn

    conversation = NeMoMultimodalConversation(
        id="conv-0",
        turns=[
            TextTurn(value="speak to me", role="user"),
            AudioTurn(cut=Recording.from_file("/path/to/audio.wav").to_cut(), role="assistant", audio_locator_tag="[audio]"),
        ],
        token_equivalent_duration=0.08,  # this value will be auto-inserted by the dataloader
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": "/path/to/chat_{0..N}.jsonl",
                    "audio_locator_tag": "[audio]",
                },
            ],
            "use_multimodal_dataloading": True,
            "token_equivalent_duration": 0.08,
            "prompt_format": "llama2",
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**Dataloading and bucketing of text and multimodal data.** When dataloading text or multimodal data, pay attention to the following config options (we provide example values for convenience):

* ``use_multimodal_sampling: true`` tells Lhotse to switch from measuring audio duration to measuring token counts; required for text.

* ``prompt_format: "prompt-name"`` will apply a specified PromptFormatter during data sampling to accurately reflect its token counts.

* ``measure_total_length: true`` customizes length measurement for decoder-only and encoder-decoder models. Decoder-only models consume a linear sequence of context + answer, so we should measure the total length (``true``). On the other hand, encoder-decoder models deal with two different sequence lengths: input (context) sequence length for the encoder, and output (answer) sequence length for the decoder. For such models set this to ``false``.

* ``min_tokens: 1``/``max_tokens: 4096`` filters examples based on their token count (after applying the prompt format).

* ``min_tpt: 0.1``/``max_tpt: 10`` filter examples based on their output-token-per-input-token-ratio. For example, a ``max_tpt: 10`` means we'll filter every example that has more than 10 output tokens per 1 input token. Very useful for removing sequence length outliers that lead to OOM. Use ``estimate_token_bins.py`` to view token count distributions for calbirating this value.

* (multimodal-only) ``token_equivalent_duration: 0.08`` is used to be able to measure audio examples in the number of "tokens". For example, if we're using fbank with 0.01s frame shift and an acoustic model that has a subsampling factor of 0.08, then a reasonable setting for this could be 0.08 (which means every subsampled frame counts as one token). Calibrate this value to fit your needs.

**Text/multimodal bucketing and OOMptimizer.** Analogous to bucketing for audio data, we provide two scripts to support efficient bucketing:

* ``scripts/speech_llm/estimate_token_bins.py`` which estimates 1D or 2D buckets based on the input config, tokenizer, and prompt format. It also estimates input/output token count distribution and suggested ``max_tpt`` (token-per-token) filtering values.

* (experimental) ``scripts/speech_llm/oomptimizer.py`` which works with SALM/BESTOW GPT/T5 models and estimates the optimal ``bucket_batch_size`` for a given model config and bucket bins value. Given the complexity of Speech LLM some configurations may not be supported yet at the time of writing (e.g., model parallelism).

To enable bucketing, set ``batch_size: null`` and use the following options:

* ``use_bucketing: true``

* ``bucket_duration_bins`` - the output of ``estimate_token_bins.py``. If ``null``, it will be estimated at the start of training at the cost of some run time (not recommended).

* (oomptimizer-only) ``bucket_batch_size`` - the output of OOMptimizer.

* (non-oomptimizer-only) ``batch_tokens`` is the maximum number of tokens we want to find inside a mini-batch. Similarly to ``batch_duration``, this number does consider padding tokens too, therefore enabling bucketing is recommended to maximize the ratio of real vs padding tokens. Note that it's just a heuristic for determining the optimal batch sizes for different buckets, and may be less efficient than using OOMptimizer.

* (non-oomptimizer-only) ``quadratic_factor`` is a quadratic penalty to equalize the GPU memory usage between buckets of short and long sequence lengths for models with quadratic memory usage. It is only a heuristic and may not be as efficient as using OOMptimizer.

**Joint dataloading of text/audio/multimodal data.** The key strength of this approach is that we can easily combine audio datasets and text datasets,
and benefit from every other technique we described in this doc, such as: dynamic data mixing, data weighting, dynamic bucketing, and so on.

This approach is described in the `EMMeTT`_ paper. There's also a notebook tutorial called Multimodal Lhotse Dataloading. We construct a separate sampler (with its own batching settings) for each modality,
and specify how the samplers should be fused together via the option ``sampler_fusion``:

* ``sampler_fusion: "round_robin"`` will iterate single sampler per step, taking turns. For example: step 0 - audio batch, step 1 - text batch, step 2 - audio batch, etc.

* ``sampler_fusion: "randomized_round_robin"`` is similar, but at each chooses a sampler randomly using ``sampler_weights: [w0, w1]`` (weights can be unnormalized).

* ``sampler_fusion: "zip"`` will draw a mini-batch from each sampler at every step, and merge them into a single ``CutSet``. This approach combines well with multimodal gradient accumulation (run forward+backward for one modality, then the other, then the update step).

.. _EMMeTT: https://arxiv.org/abs/2409.13523

Example. Combine an ASR (audio-text) dataset with an MT (text-only) dataset so that mini-batches have some examples from both datasets:

.. code-block:: yaml

    model:
      ...
      train_ds:
        multi_config: True,
        sampler_fusion: zip
        shuffle: true
        num_workers: 4

        audio:
          prompt_format: t5nmt
          use_bucketing: true
          min_duration: 0.5
          max_duration: 30.0
          max_tps: 12.0
          bucket_duration_bins: [[3.16, 10], [3.16, 22], [5.18, 15], ...]
          bucket_batch_size: [1024, 768, 832, ...]
          input_cfg:
            - type: nemo_tarred
              manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
              tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
              weight: 0.5
              tags:
                context: "Translate the following to English"

        text:
          prompt_format: t5nmt
          use_multimodal_sampling: true
          min_tokens: 1
          max_tokens: 256
          min_tpt: 0.333
          max_tpt: 3.0
          measure_total_length: false
          use_bucketing: true
          bucket_duration_bins: [[10, 4], [10, 26], [15, 10], ...]
          bucket_batch_size: [512, 128, 192, ...]
          input_cfg:
            - type: txt_pair
              source_path: /path/to/en__OP_0..512_CL_.txt
              target_path: /path/to/pl__OP_0..512_CL_.txt
              source_language: en
              target_language: pl
              weight: 0.5
              tags:
                question: "Translate the following to Polish"

.. caution:: We strongly recommend to use multiple shards for text files as well so that different nodes and dataloading workers are able to randomize the order of text iteration. Otherwise, multi-GPU training has a high risk of duplication of text examples.

Pre-computing bucket duration bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend to pre-compute the bucket duration bins in order to accelerate the start of the training -- otherwise, the dynamic bucketing sampler will have to spend some time estimating them before the training starts.
The following script may be used:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 manifest.json

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.78,2.34,2.69,...
    <other diagnostic information about the dataset>

For multi-dataset setups, one may provide a dataset config directly:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 input_cfg.yaml

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.91,3.02,3.56,...
    <other diagnostic information about the dataset>

It's also possible to manually specify the list of data manifests (optionally together with weights):

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 [[manifest.json,0.7],[other.json,0.3]]

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.91,3.02,3.56,...
    <other diagnostic information about the dataset>

2D bucketing
~~~~~~~~~~~~

To achieve maximum training efficiency for some classes of models it is necessary to stratify the sampling
both on the input sequence lengths and the output sequence lengths.
One such example are attention encoder-decoder models, where the overall GPU memory usage can be factorized
into two main components: input-sequence-length bound (encoder activations) and output-sequence-length bound
(decoder activations).
Classical bucketing techniques only stratify on the input sequence length (e.g. duration in speech),
which leverages encoder effectively but leads to excessive padding on on decoder's side.

To amend this we support a 2D bucketing technique which estimates the buckets in two stages.
The first stage is identical to 1D bucketing, i.e. we determine the input-sequence bucket bins so that
every bin holds roughly an equal duration of audio.
In the second stage, we use a tokenizer and optionally a prompt formatter (for prompted models) to
estimate the total number of tokens in each duration bin, and sub-divide it into several sub-buckets,
where each sub-bucket again holds roughly an equal number of tokens.

To run 2D bucketing with 30 buckets sub-divided into 5 sub-buckets each (150 buckets total), use the following script:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --tokenizer path/to/tokenizer.model \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

    # The script's output:
    Use the following options in your config:
            use_bucketing=1
            num_buckets=30
            bucket_duration_bins=[[1.91,10],[1.91,17],[1.91,25],...
    The max_tps setting below is optional, use it if your data has low quality long transcript outliers:
            max_tps=[13.2,13.2,11.8,11.8,...]

Note that the output in ``bucket_duration_bins`` is a nested list, where every bin specifies
the maximum duration and the maximum number of tokens that go into the bucket.
Passing this option to Lhotse dataloader will automatically enable 2D bucketing.

Note the presence of ``max_tps`` (token-per-second) option.
It is optional to include it in the dataloader configuration: if you do, we will apply an extra filter
that discards examples which have more tokens per second than the threshold value.
The threshold is determined for each bucket separately based on data distribution, and can be controlled
with the option ``--token_outlier_threshold``.
This filtering is useful primarily for noisy datasets to discard low quality examples / outliers.

We also support aggregate tokenizers for 2D bucketing estimation:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --tokenizer path/to/en/tokenizer.model path/to/pl/tokenizer1.model \
        --langs en pl \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

To estimate 2D buckets for a prompted model such as Canary-1B, provide prompt format name and an example prompt.
For Canary-1B, we'll also provide the special tokens tokenizer. Example:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --prompt-format canary \
        --prompt "[{'role':'user','slots':{'source_lang':'en','target_lang':'de','task':'ast','pnc':'yes'}}]" \
        --tokenizer path/to/spl_tokens/tokenizer.model path/to/en/tokenizer.model path/to/de/tokenizer1.model \
        --langs spl_tokens en de \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

Pushing GPU utilization to the limits with bucketing and OOMptimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default approach of specifying a ``batch_duration``, ``bucket_duration_bins`` and ``quadratic_duration``
is quite flexible, but is not maximally efficient. We observed that in practice it often leads to under-utilization
of GPU memory and compute for most buckets (especially those with shorter durations).
While it is impossible to estimate GPU memory usage up-front, we can determine it empirically with a bit of search.

OOMptimizer is an approach that given a NeMo model, optimizer, and a list of buckets (1D or 2D)
estimates the maximum possible batch size to use for each bucket.
It performs a binary search over batch sizes that succeed or lead to CUDA OOM until convergence.
We find that the resulting bucketing batch size profiles enable full GPU utilization in training,
while it only takes a couple of minutes to complete the search.

In order to run OOMptimizer, you only need the bucketing bins (from previous sections) and a model configuration:

.. code-block:: bash

    $ python scripts/speech_recognition/oomptimizer.py \
        --config-path fast-conformer_aed.yaml \
        --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
        --buckets '[[3.975,30],[3.975,48],[4.97,37],[4.97,60],[5.851,42],[5.851,71],[6.563,46],[6.563,79],[7.32,49],[7.32,88],[8.19,54],[8.19,99],[8.88,61],[8.88,107],[9.75,66],[9.75,117],[10.55,72],[10.55,127],[11.21,76],[11.21,135],[11.87,79],[11.87,143],[12.54,82],[12.54,151],[13.08,87],[13.08,157],[13.62,91],[13.62,164],[14.16,93],[14.16,170],[14.7,96],[14.7,177],[15.19,99],[15.19,183],[15.67,101],[15.67,189],[16.13,103],[16.13,194],[16.66,105],[16.66,200],[17.2,108],[17.2,207],[17.73,111],[17.73,213],[18.2,114],[18.2,219],[18.69,117],[18.69,225],[19.15,120],[19.15,230],[19.62,123],[19.62,236],[20.264,122],[20.264,244],[32.547,173],[32.547,391],[36.587,227],[36.587,440],[40.0,253],[40.0,480]]'

    # The script's output:
    <output logs from the search>
    The final profile is:
            bucket_duration_bins=[[3.975,30],[3.975,48],[4.97,37],[4.97,60],[5.851,42],[5.851,71],[6.563,46],[6.563,79],[7.32,49],[7.32,88],[8.19,54],[8.19,99],[8.88,61],[8.88,107],[9.75,66],[9.75,117],[10.55,72],[10.55,127],[11.21,76],[11.21,135],[11.87,79],[11.87,143],[12.54,82],[12.54,151],[13.08,87],[13.08,157],[13.62,91],[13.62,164],[14.16,93],[14.16,170],[14.7,96],[14.7,177],[15.19,99],[15.19,183],[15.67,101],[15.67,189],[16.13,103],[16.13,194],[16.66,105],[16.66,200],[17.2,108],[17.2,207],[17.73,111],[17.73,213],[18.2,114],[18.2,219],[18.69,117],[18.69,225],[19.15,120],[19.15,230],[19.62,123],[19.62,236],[20.264,122],[20.264,244],[32.547,173],[32.547,391],[36.587,227],[36.587,440],[40.0,253],[40.0,480]]
            bucket_batch_size=[352,308,280,245,245,206,206,180,186,163,168,142,151,132,136,119,126,106,116,98,110,92,104,88,99,83,94,79,90,76,86,72,86,72,81,68,80,65,78,63,74,60,72,58,70,58,68,54,66,52,65,52,62,50,37,28,31,24,28,21]
            max_tps=12.0
            max_duration=40.0

Use the resulting options in your training configuration (typically under namespace ``model.train_ds``) to apply the profile.

It's also possible to run OOMptimizer using a pretrained model's name and bucket bins corresponding
to your fine-tuning data:

    $ python scripts/speech_recognition/oomptimizer.py \
        --pretrained-name nvidia/canary-1b \
        --buckets '[2.0,3.1,5.6,6.6,...]'

Note that your training script can perform some additional actions using GPU RAM that cannot be anticipated by the OOMptimizer.
By default, we let the script use up to 90% of GPU's RAM for this estimation to account for that.
In the unlikely case you run into an OutOfMemoryError during training, you can try re-estimating the profile with the option ``--memory-fraction 0.75`` (or another value) that will further cap OOMptimizer's available GPU RAM.

Seeds and randomness
~~~~~~~~~~~~~~~~~~~~

In Lhotse dataloading configuration we have two parameters controlling randomness: ``seed`` and ``shard_seed``.
Both of them can be either set to a fixed number, or one of two string options ``"randomized"`` and ``"trng"``.
Their roles are:

* ``seed`` is the base random seed, and is one of several factors used to initialize various RNGs participating in dataloading.

* ``shard_seed`` controls the shard randomization strategy in distributed data parallel setups when using sharded tarred datasets.

Below are the typical examples of configuration with an explanation of the expected outcome.

Case 1 (default): ``seed=<int>`` and ``shard_seed="trng"``:

* The ``trng`` setting discards ``seed`` and causes the actual random seed to be drawn using OS's true RNG. Each node/GPU/dataloading worker draws its own unique random seed when it first needs it.

* Each node/GPU/dataloading worker yields data in a different order (no mini-batch duplication).

* On each training script run, the order of dataloader examples are **different**.

* Since the random seed is unpredictable, the exact dataloading order is not replicable.

Case 2: ``seed=<int>`` and ``shard_seed="randomized"``:

* The ``randomized`` setting uses ``seed`` along with DDP ``rank`` and dataloading ``worker_id`` to set a unique but deterministic random seed in each dataloading process across all GPUs.

* Each node/GPU/dataloading worker yields data in a different order (no mini-batch duplication).

* On each training script run, the order of dataloader examples are **identical** as long as ``seed`` is the same.

* This setup guarantees 100% dataloading reproducibility.

* Resuming training without changing of the ``seed`` value will cause the model to train on data it has already seen. For large data setups, not managing the ``seed`` may cause the model to never be trained on a majority of data. This is why this mode is not the default.

* If you're combining DDP with model parallelism techniques (Tensor Parallel, Pipeline Parallel, etc.) you need to use ``shard_seed="randomized"``. Using ``"trng"`` will cause different model parallel ranks to desynchronize and cause a deadlock.

* Generally the seed can be managed by the user by providing a different value each time the training script is launched. For example, for most models the option to override would be ``model.train_ds.seed=<value>``. If you're launching multiple tasks queued one after another on a grid system, you can generate a different random seed for each task, e.g. on most Unix systems ``RSEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')`` would generate a random uint32 number that can be provided as the seed.

Other, more exotic configurations:

* With ``shard_seed=<int>``, all dataloading workers will yield the same results. This is only useful for unit testing and maybe debugging.

* With ``seed="trng"``, the base random seed itself will be drawn using a TRNG. It will be different on each GPU training process. This setting is not recommended.

* With ``seed="randomized"``, the base random seed is set to Python's global RNG seed. It might be different on each GPU training process. This setting is not recommended.

Preparing Text-Only Data for Hybrid ASR-TTS Models
--------------------------------------------------

:ref:`Hybrid ASR-TTS models <Hybrid-ASR-TTS_model>` require a text-only dataset for training the ASR model.
Each record in the dataset (in ``.json`` file) should contain the following fields:

* ``text``: text to use as a target for the ASR model
* ``tts_text`` or/and ``tts_text_normalized``: text to use as a source for TTS model. ``tts_text_normalized`` should contain normalized text for TTS model. If there is no such field, ``tts_text`` will be used after normalization using the normalizer from the TTS model. It is highly recommended to normalize the text and create ``tts_text_normalized`` field manually, since current normalizers are unsuitable for processing a large amount of text on the fly.

**Example record:**

.. code-block:: json

    {"text": "target for one hundred billion parameters asr model",
     "tts_text": "Target for 100B parameters ASR model.",
     "tts_text_normalized": "Target for one hundred billion parameters ASR model."}
