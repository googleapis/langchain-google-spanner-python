# -*- coding: utf-8 -*-
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import os
import pathlib
from pathlib import Path
import shutil

import nox

DEFAULT_PYTHON_VERSION = "3.10"
CURRENT_DIRECTORY = pathlib.Path(__file__).parent.absolute()
LINT_PATHS = ["src", "tests", "noxfile.py"]


nox.options.sessions = [
    "blacken",
    "docfx",
    "docs",
    "format",
    "lint",
    "unit",
]

# Error if a python version is missing
nox.options.error_on_missing_interpreters = True


@nox.session(python=DEFAULT_PYTHON_VERSION)
def docs(session):
    """Build the docs for this library."""

    session.install("-e", ".")
    session.install(
        # We need to pin to specific versions of the `sphinxcontrib-*` packages
        # which still support sphinx 4.x.
        # See https://github.com/googleapis/sphinx-docfx-yaml/issues/344
        # and https://github.com/googleapis/sphinx-docfx-yaml/issues/345.
        "sphinxcontrib-applehelp==1.0.4",
        "sphinxcontrib-devhelp==1.0.2",
        "sphinxcontrib-htmlhelp==2.0.1",
        "sphinxcontrib-qthelp==1.0.3",
        "sphinxcontrib-serializinghtml==1.1.5",
        "sphinx==4.5.0",
        "alabaster",
        "recommonmark",
    )

    shutil.rmtree(os.path.join("docs", "_build"), ignore_errors=True)
    session.run(
        "sphinx-build",
        "-W",  # warnings as errors
        "-T",  # show full traceback on exception
        "-N",  # no colors
        "-b",
        "html",
        "-d",
        os.path.join("docs", "_build", "doctrees", ""),
        os.path.join("docs", ""),
        os.path.join("docs", "_build", "html", ""),
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def docfx(session):
    """Build the docfx yaml files for this library."""

    session.install("-e", ".")
    session.install(
        # We need to pin to specific versions of the `sphinxcontrib-*` packages
        # which still support sphinx 4.x.
        # See https://github.com/googleapis/sphinx-docfx-yaml/issues/344
        # and https://github.com/googleapis/sphinx-docfx-yaml/issues/345.
        "sphinxcontrib-applehelp==1.0.4",
        "sphinxcontrib-devhelp==1.0.2",
        "sphinxcontrib-htmlhelp==2.0.1",
        "sphinxcontrib-qthelp==1.0.3",
        "sphinxcontrib-serializinghtml==1.1.5",
        "gcp-sphinx-docfx-yaml",
        "alabaster",
        "recommonmark",
    )

    shutil.rmtree(os.path.join("docs", "_build"), ignore_errors=True)
    session.run(
        "sphinx-build",
        "-T",  # show full traceback on exception
        "-N",  # no colors
        "-D",
        (
            "extensions=sphinx.ext.autodoc,"
            "sphinx.ext.autosummary,"
            "docfx_yaml.extension,"
            "sphinx.ext.intersphinx,"
            "sphinx.ext.coverage,"
            "sphinx.ext.napoleon,"
            "sphinx.ext.todo,"
            "sphinx.ext.viewcode,"
            "recommonmark"
        ),
        "-b",
        "html",
        "-d",
        os.path.join("docs", "_build", "doctrees", ""),
        os.path.join("docs", ""),
        os.path.join("docs", "_build", "html", ""),
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def lint(session):
    """Run linters.

    Returns a failure if the linters find linting errors or
    sufficiently serious code quality issues.
    """
    session.install(".[lint]")
    session.run(
        "black",
        "--check",
        *LINT_PATHS,
    )
    session.run("flake8", "src", "tests")


@nox.session(python=DEFAULT_PYTHON_VERSION)
def lint_setup_py(session):
    """Verify that setup.py is valid (including an RST check)."""
    session.install("docutils", "pygments")
    session.run("python", "setup.py", "check", "--restructuredtext", "--strict")


@nox.session(python=DEFAULT_PYTHON_VERSION)
def blacken(session):
    session.install(".[lint]")
    session.run(
        "black",
        *LINT_PATHS,
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def format(session):
    session.install(".[lint]")
    # Sort imports in strict alphabetical order.
    session.run(
        "isort",
        *LINT_PATHS,
    )
    session.run(
        "black",
        *LINT_PATHS,
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def unit(session):
    install_unittest_dependencies(session)
    session.run(
        "py.test",
        "--quiet",
        os.path.join("tests", "unit"),
        *session.posargs,
    )


def install_unittest_dependencies(session, *constraints):
    session.install(".[test]")
    session.run(
        "pip",
        "install",
        "--no-compile",  # To ensure no byte recompliation which is usually super slow
        "-q",
        "--disable-pip-version-check",  # Avoid the slow version check
        ".",
        "-r",
        "requirements.txt",
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def integration(session):
    install_integrationtest_dependencies(session)
    session.run(
        "py.test",
        "--quiet",
        os.path.join("tests", "integration"),
        *session.posargs,
    )


def install_integrationtest_dependencies(session):
    session.install(".[test]")
    session.run(
        "pip",
        "install",
        "--no-compile",  # To ensure no byte recompliation which is usually super slow
        "-q",
        "--disable-pip-version-check",  # Avoid the slow version check
        ".",
        "-r",
        "requirements.txt",
    )
