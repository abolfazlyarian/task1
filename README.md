# Taaghche

Taaghche is a project that aims to answer the question of what is the appropriate ranking of a given list of ten books for a specific user. It utilizes a dataset containing user interactions with books within a specified time range, as well as information about the books themselves. The project tackles the problem of learning to rank, which refers to the algorithms designed to rank items based on their relevance to a specific topic. In this case, the goal is to find a model that can predict the ranking of books for a user based on their relevance.

## Project Structure

The Taaghche project has the following directory structure:

```
taaghche
├── dataset
│ ├── actions.csv
│ └── book_data.csv
├── dockerfile
│ ├── Dockerfile.extractor
│ ├── Dockerfile.model
│ ├── Dockerfile.model-provider
│ └── Dockerfile.provider
├── data_extractor.py
├── data_provider.py
├── docker-compose2.yaml
├── docker-compose.yaml
├── model.pkl
├── model_provider.py
├── model.py
├── nginx.conf
├── README.md
├── requirements_data.txt
└── requirements_model.txt
```

The project consists of the following components:

- `data_extractor.py`: This component is responsible for extracting data from an external source. It assumes the input data is stored in CSV format on the disk. The extracted data is then validated and stored in a database.
- `data_provider.py`: This component interacts with the model component. It retrieves the necessary dataset from the databases based on the model's request.
- `dataset`: This directory contains the dataset used for user-book interactions (`actions.csv`) and book information (`book_data.csv`).
- `docker-compose2.yaml` and `docker-compose.yaml`: These files define the Docker Compose configurations for the project.
- `dockerfile`: This directory contains Dockerfiles for different components of the project, including the extractor, model, model provider, and provider.
- `model.pkl`: This file represents the main model used in the project.
- `model_provider.py`: This component utilizes the model and provides the necessary output for the system.
- `model.py`: This file contains the implementation of the main model.
- `nginx.conf`: This file defines the configuration for the Nginx server used in the project.
- `README.md`: This file (the current file you are reading) provides an overview and documentation of the Taaghche project.
- `requirements_data.txt`: This file lists the dependencies required for the data-related components.
- `requirements_model.txt`: This file lists the dependencies required for the model-related components.

## System Architecture

After designing a suitable model to solve the problem, the Taaghche system is composed of the following five components that communicate with each other using a microservices architecture:

1. **Data Extractor**: This component is responsible for reading the dataset from an external source. It simplifies the problem by assuming that the input sources are currently only available on disk in CSV format. The read data is then validated and stored in a database.

2. **MongoDB**: This is a MongoDB database where the dataset is stored.

3. **Data Provider**: This component is used by the model component. It retrieves the dataset from the databases based on the model's request.

4. **Model**: This is the main model provided by the project. It ranks the list of books based on their relevance to the user.

5. **API Gateway**: To enable output retrieval from the system, a REST API should be provided.

## Dataset

The dataset containing user-book interactions and book information is included in a compressed file you received via email. It consists of two parts:

**User-Book Interaction Data**:

| Column      | Description                        |
|-------------|------------------------------------|
| AccountId   | Unique identifier of the user       |
| BookId      | Unique identifier of the book       |
| CreationDate| Date and time of the interaction    |

**Book Information Data**:

| Column           | Description                              |
|------------------|------------------------------------------|
| book_id          | Unique identifier of the book             |
| title            | Book title                               |
| description      | Book description                         |
| price            | Book price (in Iranian Rial)              |
| number_of_pages  | Number of pages in the book               |
| PhysicalPrice    | Price of the physical copy of the book    |
| publishDate      | Publication date of the book              |
| rating           | Average rating given to the book by users |
| publisher        | Publisher of the book                     |
| categories       | Book categories                          |
| author_name      | Author of the book                        |
| translator_name  | Translator of the book (if applicable)    |
| lang             | Language of the book                      |

## Evaluation

There are several metrics available to evaluate the performance of machine learning models in the field of learning to rank (LTR). One of the most commonly used metrics is the Normalized Discounted Cumulative Gain (NDCG). NDCG measures the quality of a ranked list of recommendations based on the order and relevance of the recommendations to a specific user.

While NDCG is a suggested metric, you can choose any other metric of your preference to evaluate the model and proceed accordingly.

## Important Notes

Here are some notable points to consider regarding the Taaghche project:

- All components should be containerized using Docker and be independent of each other.
- There are no specific restrictions on the methods and libraries used.
- Input datasets should be validated before storing them in databases, ensuring that their schemas match the predefined schemas.
- Data models in the database should be designed, and code for creation and storage should be available.
- The final system should be capable of responding in a production environment. For example, assume that the system should be able to handle ten open connections, where each connection corresponds to a thread making requests, within a reasonable time frame (less than 500ms).
- Using the book information dataset is not mandatory.

Feel free to modify and enhance the project according to your requirements and preferences.
