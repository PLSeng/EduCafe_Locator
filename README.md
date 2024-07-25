# EduCafe Locator Project

## Project Overview
The EduCafe Locator project utilizes k-means clustering to identify optimal locations for new cafes in Phnom Penh, targeting areas near educational buildings. By analyzing geographical data obtained from a Google API, this project aims to strategically position cafes to maximize accessibility for students and faculty, enhancing profitability and customer retention.

## Dataset
The dataset used in this project includes geographical locations and attributes of educational buildings in Phnom Penh, sourced from the Google Maps API. The data was processed and cleaned to focus on key attributes relevant to location analysis for cafes.

## Methodology
- **Data Collection**: Data was collected via the Google Maps API, focusing on educational buildings in Phnom Penh.
- **Data Preprocessing**: The dataset was cleaned and prepared, removing irrelevant data and handling missing values.
- **K-Means Clustering**: We applied k-means clustering to group educational buildings into clusters based on their geographical proximity.
- **Elbow Method**: The elbow method was used to determine the optimal number of clusters, indicating the most suitable areas for cafe placement.

## Results
Clusters were identified that represent optimal locations for establishing new cafes based on the density and distribution of educational buildings. Each cluster was analyzed to determine the potential customer base and accessibility.

## Installation
To run this project, you will need Python and several libraries which can be installed via pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Contributors
- [PAV Limseng](https://github.com/PLSeng)
- [KHUN Sithanut](https://github.com/Sithanut-Khun)
- [PEN Virak](https://github.com/PenVirak)
- [PEL Bunkhloem](https://github.com/Thecodingsamurai)
- [PEANG Rattanak](https://github.com/Peang-Rattanak)
