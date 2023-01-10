Issues & FAQ
===================================


FAQ
########

Issues 
########


Java JVM DLL not found (specific to Mac)
**********************

One user reported the following error when installing `pyspi` on a MacBook Air an M2 chip and Catalina OS:

.. code-block::

   OSError: [Errno 0] JVM DLL not found /Library/Java/JavaVirtualMachines/jdk-19.jdk/Contents/Home/lib/libjli.dylib


This issue is similar to those reported `here <https://stackoverflow.com/questions/71504214/jvm-dll-not-found-but-i-can-clearly-see-the-file>`_ and `here <https://github.com/jpype-project/jpype/issues/994>`_; it can arise from the version of OpenJDK identified as the system default. Some Java versions don't include all of the binary (DLL) files that `pyspi` looks for.

We recommend following this `helpful tutorial <https://blog.bigoodyssey.com/how-to-manage-multiple-java-version-in-macos-e5421345f6d0>`_ by Chamika Kasun to install `AdoptOpenJDK <https://adoptopenjdk.net/index.html>`_. In a nutshell, here are the steps you should run:

Install homebrew if you don't already have it:

.. code-block::

   $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


Install `jenv` as your Java version manager:

.. code-block:: 

    $ brew install jenv


Add `jenv` to your shell's configuration file (e.g. `.bashrc` if you use `bash`):

.. code-block:: 

    $ export PATH="$HOME/.jenv/bin:$PATH"
    $ eval "$(jenv init -)"

Source your shell's configuration file:

.. code-block:: 

    $ source ~/.bashrc # If you use bash

Confirm proper installation of `jEnv`:

.. code-block::

    $ jenv doctor

Even if this returns some errors, as long as you see `Jenv is correctly loaded`, you're all set. We recommend using `AdoptOpenJDK` version 11, which you can install with the following command:

.. code-block:: 

    $ brew install AdoptOpenJDK/openjdk/adoptopenjdk11

Now, you will need to add your `AdoptOpenJDK` path to your `jEnv` environments. First, you can find where your jdk files are installed with the following command:

.. code-block:: 

    $ /usr/libexec/java_home -V

This will list all your installed java JDK versions. Locate the one for `AdoptOpenJDK` version 11 and paste the path:

.. code-block::

    $ jenv add <path_to_adopt_open_jdk_11>

Confirm `AdoptOpenJDK` version 11 was added to `jEnv`:

.. code-block:: 

    $ jenv versions

You can set `AdoptOpenJDK` version 11 as your global Java version with the following:

.. code-block:: 

    $ jenv global <AdoptOpenJDK version>
    $ # example:
    $ jenv global 11.0