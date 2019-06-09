.. _quickstart-mac:

================================================
Development Environment Quickstart Quide (macOS)
================================================

This quickstart guide will cover:

* setting up and maintaining a development environment, including installing compilers and SciPy dependencies;
* creating a personal fork of the SciPy repository on GitHub;
* using git to manage a local repository with development branches;
* performing an in-place build of SciPy; and 
* creating a virtual environment that adds this development version of SciPy to the Python path on macOS.

Its companion videos `Anaconda SciPy Dev: Part I (macOS)`_ and `Anaconda SciPy Dev: Part I (macOS)`_ show many of the steps being performed. This guide may diverge slightly from the videos over time with the goal of keeping this guide the simplest, up-to-date procedure.

.. note:: 

	This guide does not present the only way to set up a development environment; there are many valid choices of Python distribution, C/Fortran compiler, and installation options. The steps here can often be adapted for other choices, but we cannot provide documentation tailored for them.
	
	This guide assumes that you are starting without an existing Python 3 installation. If you already have Python 3, you might want to uninstall it first to avoid ambiguity over which Python version is being used at the command line. 

Building SciPy
--------------

(Consider following along with the companion video `Anaconda SciPy Dev: Part I (macOS)`_) 

#. Download, install, and test the latest release of the `Anaconda Distribution of Python`_. In addition to the latest version of Python 3, the Anaconda distribution includes dozens of the most popular Python packages for scientific computing, the Spyder integrated development environment (IDE), the ``conda`` package manager, and tools for managing virtual environments. 

#. Install Apple Developer Tools. An easy way to do this is to `open a terminal window <https://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_, enter the command ``xcode-select --install``, and follow the prompts. Apple Developer Tools includes `git <https://git-scm.com/>`_, the software we need to download and manage the SciPy source code.

#. Browse to the `SciPy repository on GitHub <https://github.com/scipy/scipy>`_ and `create your own fork <https://help.github.com/en/articles/fork-a-repo>`_. You'll need to create a GitHub account if you don't already have one.

#. Browse to your fork. Your fork will have a URL like `https://github.com/mdhaber/scipy <https://github.com/mdhaber/scipy>`_, except with your GitHub username in place of "mdhaber".

#. Click the big, green "Clone or download" button, and copy the ".git" URL to the clipboard. The URL will be the same as your fork's URL, except it will end in ".git".

#. Create a folder for the SciPy source code in a convenient place on your computer. `Navigate to it in the terminal <https://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_.

#. Enter the command ``git clone`` followed by your fork's .git URL.
Note that this creates in the terminal's working directory a ``scipy`` folder containing the SciPy source code.

#. In the terminal, navigate into the ``scipy`` root directory (e.g. ``cd scipy``).

#. Install `Homebrew`_. Enter into the terminal |br| ``/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`` |br| or follow the installation instructions listed on the Homebrew website. 
Homebrew is a package manager for macOS that will help you download `gcc`, the software we will use to compile C, C++, and Fortran code included in SciPy.


.. _Anaconda SciPy Dev\: Part I (macOS): https://youtu.be/1rPOSNd0ULI

.. _Anaconda SciPy Dev\: Part II (macOS): https://youtu.be/Faz29u5xIZc

.. _Anaconda Distribution of Python: https://www.anaconda.com/distribution/

.. _Homebrew: https://brew.sh/

.. |br| raw:: html

    <br>
