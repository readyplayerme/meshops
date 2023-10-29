# Welcome to the Ready Player Me Texture Synthesis contributing guide <!-- omit in toc -->

In this guide you will get an overview of the contribution workflow from opening an issue, creating a pull request, reviewing, and merging the pull request.

## New contributor guide

To get an overview of the project, read the [README](README.md).
Here are some resources to help you get started with open source contributions:

- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)
- Learn about [pre-commit hooks](https://pre-commit.com/)
- We use [black](https://black.readthedocs.io/en/stable/) formatting, but with a line-length of 120 characters.
- If you haven't yet setup an IDE, we recommend [Visual Studio Code](https://code.visualstudio.com/). See [Python in Visual Studio Code](https://code.visualstudio.com/docs/languages/python).

## Issues

### Create a new issue

If you spot a problem with the schemas or package, [search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments).
If a related issue doesn't exist, you can open a new issue using a relevant [issue form](https://github.com/readyplayerme/texturesynthesis/issues/new/choose).

### Solve an issue

Scan through our [existing issues](https://github.com/readyplayerme/texturesynthesis/issues) to find one that interests you.
You can narrow down the search using `labels` as filters.

### Labels

Labels can help you find an issue you'd like to help with.
The [`good first issue` label](https://github.com/readyplayerme/texturesynthesis/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) is for problems or updates we think are ideal for new joiners.

## Contribute

### Get the Code

1. If you are a contributor from outside the Ready Player Me team, [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

2. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the (forked) repository to your local machine.

### Setting up Development Environment

It's best to use a separate Python _environment_ for development to avoid conflicts with other Python projects and keep your system Python clean. In this section we'll provide instructions on how to set up such an environment.

We use [hatch](https://hatch.pypa.io/) as the Python package build backend and Python project manager.
We recommend to install it as it will provide you with a project-specific development environment. However, using hatch is not a necessity, but more of a convenience.  
Unfortunately, there are no pre-built binaries for hatch, and hatch on its own can only create environments with Python versions that are already installed on your system. So you'll need to first create a Python environment to install hatch into, in order to then spawn another environment for the project by using hatch. It's a bit like the chicken & egg problem paired with the movie Inception.😵‍💫 We'll walk you through it.

1. We encourage using an environment manager such as [conda](https://docs.conda.io/en/latest/), [mamba/micromamba](https://mamba.readthedocs.io/en/latest/index.html), or [poetry](https://python-poetry.org/).  
    You'll need a minimum Python version 3.10.
    Here's an example on Windows:

    ```powershell
    # Get mamba using winget.
    winget install -e --id CondaForge.Mambaforge

    # Make mamba available in your shell. mamba may be either installed in %ProgramData% or %LocalAppData%.
    %ProgramData%\mambaforge\.condabin\mamba init
    # OR, if your mamba installation is in %LocalAppData% instead:
    %LocalAppData%\mambaforge\.condabin\mamba init
    # You may need to restart your terminal now.

    # Test that mamba is available.
    mamba --version  # This should print something like "mamba 1.4.1".
    ```

2. You can read [hatch's installation instructions](https://hatch.pypa.io/latest/install/) for information on how to install it into your Python environment, or follow the instructions below.

    If you use conda/mamba, you can create a Python environment to which hatch gets installed with:

    ```powershell
    mamba create -n hatch python=3.10 hatch
    ```

    In the command above, the `-n hatch` option just gives the environment the name _hatch_, but it can be anything.
    The name _hatch_ for the environment was incidentally chosen here to match the 1 package we want to utilize in this environment. The `python=3.10 hatch` part of the command defines what we want to install into the environment. See [mamba's documentation](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart) for more details.

3. Activate the hatch environment.

    ```powershell
    mamba activate hatch
    ```

    OR if you're using Powershell (see [issue](https://github.com/mamba-org/mamba/issues/1717)):

    ```powershell
    conda activate hatch
    ```

4. Prepare the environment for development.
    Once you setup hatch, navigate to the cloned repository, and execute:

    ```powershell
    hatch env create
    ```

    This will create yet a new environment within a `.venv` folder of the project and install the development dependencies into it.
    An IDE like [Visual Studio Code](https://code.visualstudio.com/) will automatically detect this environment and suggest to you to use it as the Python interpreter for this repository.
    It also installs [pre-commit](https://pre-commit.com/) hooks into the repository, which will run linting and formatting checks before you commit your changes.

    Alternatively, you can get the new environment path and add it to your IDE as a Python interpreter for this repository with:

    ```powershell
    hatch run python -c "import sys;print(sys.executable)"
    ```

    If you decided against using hatch, we still recommend installing the pre-commit hooks.

    ```powershell
    pre-commit install -t pre-commit -t commit-msg -t pre-push
    ```

### Branch Off & Make Your Changes

1. Create a working branch and prefix its name with _fix/_ if it's a bug fix, or _feature/_ if it's a new feature.
    Start with your changes!  
    Have a look at the README for more information on how to use the package.

2. Write or update tests for your changes. <!-- TODO Explain how we do tests -->

3. Run tests with `hatch run test` and or run linting & formatting and tests with `hatch run all` locally.

### Commit your update

Once you are happy with your changes, it's time to commit them.
Use [Conventional Commit messages](https://www.conventionalcommits.org/en/v1.0.0/).  
[Sign](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits) your commits!

If you followed the steps above, you should have a pre-commit hook installed that will automatically run the tests and linting before a commit succeeds.

Keep your individual commits small, so any breaking change can more easily be traced back to a commit.
A commit ideally only changes one single responsibility at a time.
If you keep the whole of your changes small and the branch short-lived, there's less risk to run into any other conflicts that can arise with the base.

Don't forget to [self-review](#self-review) to speed up the review process :zap:.

### Pull Request

When you're finished with the changes, create a __draft__ pull request, also known as a PR.

- Fill the "Ready for review" template so that we can review your PR. This template helps reviewers understand your changes as well as the purpose of your pull request.
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- If you are a contributor from outside the Ready Player Me team, enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a Ready Player Me team member will review your proposal.
We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments.
You can apply suggested changes directly through the UI.
You can make any other changes in your branch, and then commit them.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Self review

You should always review your own PR first.

Make sure that you:

- [ ] Confirm that the changes meet the user experience and goals outlined in the design plan (if there is one).
- [ ] Update the version of the schemas.
- [ ] Update the documentation if necessary.
- [ ] If there are any failing checks in your PR, troubleshoot them until they're all passing.

### Merging your PR

Once your PR has the required approvals, a Ready Player Me team member will merge it.

We use a [squash & merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits) by default to merge a PR.

The branch will be deleted automatically after the merge to prevent any more work being done the branch after it was merged.
