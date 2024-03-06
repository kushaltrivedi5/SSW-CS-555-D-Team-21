# SSW-CS-555-D-Team-21 - AgiSynergy EEG project

# Architecture and Design Of The Web Application, Including User Interface Mockups

Architecture:

Client-Server Model: Web application follows a client-server model where the browser (client) communicates with the server to fetch data and perform actions.
Frontend: Developed using HTML, CSS, and JavaScript with React.js for dynamic user interfaces.
Backend: Built with Node.js or Python, utilizing frameworks like Express.js or Django. Handles client requests, interacts with the database, and performs business logic.
Database: Relational databases such as MySQL or PostgreSQL store structured data, including user information and application data.
APIs: RESTful APIs define endpoints and data formats for communication between frontend and backend.
Authentication and Authorization: Implements user authentication and authorization using techniques like JSON Web Tokens (JWT) or session-based authentication.
Scalability and Performance: Designed for scalability and performance with techniques like load balancing, caching, and horizontal scaling.
Design:

User Interface (UI) Design: Modern design principles ensure a clean and intuitive layout, focusing on usability and accessibility.
Responsive Design: Ensures seamless experience across various devices and screen sizes.
Color Scheme and Branding: Consistent with branding guidelines for a visually appealing interface.
Navigation: Simple and intuitive navigation with clear labels and organized menus for easy user navigation.
Forms and Input Fields: Designed for usability with clear instructions and feedback for users.
Error Handling: Provides helpful error messages and feedback for resolving issues.
Mockups: Illustrative mockups for key pages such as the homepage, user profile page, and settings page.

# EEG Dataset and Code Explanation

This repository contains code for analyzing EEG (Electroencephalography) signals for two different tasks: Few-shot EEG learning and Imagined speech classification. Additionally, it includes a dataset downloaded from the [provided link](https://osf.io/pq7vb/), which consists of EEG recordings for these tasks.

## User Login Code Explanation

In addition to the EEG analysis code, this repository includes a test file for user login to the EEG database. It's important to note that the user login code provided here is still stored locally and has not been connected to the cloud.

The user login functionality is implemented using HTML, CSS, and TypeScript, along with backend validation handled by the `testcases.py` file.

- **HTML and CSS**: The HTML and CSS files provide the structure and styling for the user login page, ensuring a user-friendly interface for logging into the EEG database.

- **TypeScript**: TypeScript is used to enhance the functionality of the user login page by adding client-side validation and handling user interactions, such as form submissions and error messages.

- **testcases.py**: This Python file serves as the backend for user authentication. It includes functions to validate user credentials, such as username and password, against the EEG database. Additionally, it handles user authentication requests and provides appropriate responses based on the validation results.

# User Story: Easy Navigation and Feature Understanding for EEG Speech Classification
To ensure the EEG speech classification web application meets user needs and acceptance criteria, here's a summary of the plan:

Clear Navigation:

Implement a well-organized navigation menu with descriptive labels.
Use dropdown menus or nested navigation for advanced features.
Intuitive User Interface:

Design a simple and clear interface with familiar UI elements.
Provide visual cues to indicate interactive elements.
Help/Tutorial Sections:

Include a dedicated "Help" or "Tutorial" section.
Create informative guides with step-by-step instructions and examples.
Consistent Interactions:

Ensure consistency in UI elements and layout across all pages.
Provide feedback to users for their actions.
Testing and Feedback:

Conduct usability testing with target users.
Gather feedback to refine the interface iteratively.

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
