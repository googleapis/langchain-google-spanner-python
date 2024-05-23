# DEVELOPER.md

## Versioning

This library follows [Semantic Versioning](http://semver.org/).

## Processes

### Conventional Commit messages

This repository uses tool [Release Please](https://github.com/googleapis/release-please) to create GitHub and PyPi releases. It does so by parsing your
git history, looking for [Conventional Commit messages](https://www.conventionalcommits.org/),
and creating release PRs.

Learn more by reading [How should I write my commits?](https://github.com/googleapis/release-please?tab=readme-ov-file#how-should-i-write-my-commits)

## Testing

### Run tests locally

1. Set environment variables for `INSTANCE_ID`, `GOOGLE_DATABASE`, `PG_DATABASE`, `TABLE_NAME`

1. Run pytest to automatically run all tests:

    ```bash
    pytest
    ```

### CI Platform Setup

Cloud Build is used to run tests against Google Cloud resources in test project: langchain-spanner-testing.
Each test has a corresponding Cloud Build trigger, see [all triggers][triggers].
These tests are registered as required tests in `.github/sync-repo-settings.yaml`.

#### Trigger Setup

Cloud Build triggers (for Python versions 3.8 to 3.11) were created with the following specs:

```YAML
name: integration-test-pr-py38
description: Run integration tests on PR for Python 3.8
filename: integration.cloudbuild.yaml
github:
  name: langchain-google-spanner-python
  owner: googleapis
  pullRequest:
    branch: .*
    commentControl: COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY
ignoredFiles:
  - docs/**
  - .kokoro/**
  - .github/**
  - "*.md"
substitutions:
  _INSTANCE_ID: <ADD_VALUE>
  _PG_DATABASE: <ADD_VALUE>
  _GOOGLE_DATABASE: <ADD_VALUE>
  _VERSION: "3.8"
```

Use `gcloud builds triggers import --source=trigger.yaml` to create triggers via the command line

#### Project Setup

1. Create an spanner instance and database
1. Setup Cloud Build triggers (above)

#### Run tests with Cloud Build

* Run integration test:

    ```bash
    gcloud builds submit --config integration.cloudbuild.yaml --region us-central1 --substitutions=_INSTANCE_ID=$INSTANCE_ID
    ```

#### Trigger

To run Cloud Build tests on GitHub from external contributors, ie RenovateBot, comment: `/gcbrun`.

#### Code Coverage
Please make sure your code is fully tested. The Cloud Build integration tests are run with the `pytest-cov` code coverage plugin. They fail for PRs with a code coverage less than the threshold specified in `.coveragerc`.  If your file is inside the main module and should be ignored by code coverage check, add it to the `omit` section of `.coveragerc`.

Check for code coverage report in any Cloud Build integration test log. 
Here is a breakdown of the report:
- `Stmts`:  lines of executable code (statements).
- `Miss`: number of lines not covered by tests.
- `Branch`: branches of executable code (e.g an if-else clause may count as 1 statement but 2 branches; test for both conditions to have both branches covered).
- `BrPart`: number of branches not covered by tests.
- `Cover`: average coverage of files.
- `Missing`: lines that are not covered by tests.


[triggers]: https://console.cloud.google.com/cloud-build/triggers?e=13802955&project=langchain-spanner-testing
