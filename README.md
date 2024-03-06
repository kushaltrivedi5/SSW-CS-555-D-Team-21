# SSW-CS-555-D-Team-21 - AgiSynergy EEG project

# EEG Dataset and Code Explanation

This repository contains code for analyzing EEG (Electroencephalography) signals for two different tasks: Few-shot EEG learning and Imagined speech classification. Additionally, it includes a dataset downloaded from the [provided link](https://osf.io/pq7vb/), which consists of EEG recordings for these tasks.

## User Login Code Explanation

In addition to the EEG analysis code, this repository includes a test file for user login to the EEG database. It's important to note that the user login code provided here is still stored locally and has not been connected to the cloud.

The user login functionality is implemented using HTML, CSS, and TypeScript, along with backend validation handled by the `testcases.py` file.

- **HTML and CSS**: The HTML and CSS files provide the structure and styling for the user login page, ensuring a user-friendly interface for logging into the EEG database.

- **TypeScript**: TypeScript is used to enhance the functionality of the user login page by adding client-side validation and handling user interactions, such as form submissions and error messages.

- **testcases.py**: This Python file serves as the backend for user authentication. It includes functions to validate user credentials, such as username and password, against the EEG database. Additionally, it handles user authentication requests and provides appropriate responses based on the validation results.

## Note

- **Early Stage Project**: It's important to note that this repository is in the initial phase of development. There are upcoming sprints planned to further enhance the project.
- Your feedback and contributions are welcome as we continue to refine the codebase and dataset analysis. Feel free to reach out to the repository owner or contributors with any suggestions or questions.

## Dataset Description

### Track 1 (Few-shot EEG learning):

- Suitable if your audience is interested in few-shot learning algorithms and EEG signal classification.
- Highlights the specific task of classifying motor imagery tasks (left hand vs. right hand) using EEG signals.
- Provides clear details about the dataset split and performance measurement criteria (classification accuracy).

### Track 3 (Imagined speech classification):

- Relevant if your audience is interested in EEG-based speech classification or communication.
- Focuses on classifying imagined speech words/phrases for basic communication.
- Offers information about the dataset split and performance measurement criteria (classification accuracy).
