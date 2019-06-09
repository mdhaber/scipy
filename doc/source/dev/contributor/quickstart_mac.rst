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
#. Install Apple Developer Tools. 
An easy way to do this is to `open a terminal window <https://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_, enter the command ``xcode-select --install``, and follow the prompts. Apple Developer Tools includes `git <https://git-scm.com/>`_, the software we need to download and manage the SciPy source code.

.. _Anaconda SciPy Dev: Part I (macOS): https://youtu.be/1rPOSNd0ULI
.. _Anaconda SciPy Dev: Part II (macOS): https://youtu.be/Faz29u5xIZc
.. _Anaconda Distribution of Python: https://www.anaconda.com/distribution/

