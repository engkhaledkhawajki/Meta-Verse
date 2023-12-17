# Project Meta-Verse
## Overview
This repository houses the project for the Meta-Verse, an innovative platform that leverages various components to enable collaborative and secure machine learning in a distributed environment.

## Main Problem
This project is mainly about translating student's emotions to their avatars so that teacher will be engaged with students.
To solve this issue we tend to apply Federated Learning.

From this problem we reached to sub problem can be divided to:
1. Privacy: that where reverse engineering could cause an issue by retrieving the images from the weights
2. Effeciency: that happens because of the high heterogeneity of the data between students

In order to improve 1 we used Differential Privacy to add noise to weights during training so that no one will be able to retrieve the images from them.
For 2 we developed a grouping algorithm that make sure all groups have the same data distribution

## Folder Structure
1. Client_Server:
This directory contains Python (.py) files responsible for implementing client and server classes components.

2. Data:
This folder hosts a file dedicated to creating, modifying, and manipulating the overall data or individual client data.

3. Differential_Privacy:
The files in this directory implement the concept of Differential Privacy (DP) for client models' weights, ensuring privacy-preserving machine learning.

4. Inter_Clustering_Grouping:
This section showcases a custom clustering algorithm designed to accommodate various types of data, fostering effective inter-clustering and grouping.

5. Models:
Here, you'll find the implementation of a Keras model, providing a foundation for the machine learning aspects of the Meta-Verse project.

6. Visualization:
The contents of this directory focus on extracting and plotting metrics or bars that describe each cluster's/group's data, enhancing the visual representation of the project's outcomes.

7. Federated_Learning:
This section is dedicated to implementing Federated Averaging (Fed_Avg) with the components mentioned in previous sections. It showcases the collaborative learning approach employed in the Meta-Verse project.

## Code Structure
Each folder consists of classes, and within each class, functions are meticulously organized. The functions are accompanied by explanations or comments, providing clarity on their respective functionalities.

For any inquiries, issues, or questions related to the code or the ideas implemented within this repository, please don't hesitate to reach out. Your feedback and engagement are highly valued.
