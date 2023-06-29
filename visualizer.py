from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, df, predicted_label, actual_label) -> None:
        self.df = df 
        self.predicted_label = predicted_label
        self.actual_label = actual_label

    def plot (self):
        pca = PCA (n_components=2)
        reduced_data = pca.fit_transform(self.df)

        # Convert string labels to integer labels
        label_mapping = {label: index for index, label in enumerate(set(self.actual_label))}
        integer_predicted_labels = [label_mapping[label] for label in self.predicted_label]

        # Plotting the clusters
        colors = ['red', 'green', 'blue']
        for i in range(reduced_data.shape[0]):
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=colors[integer_predicted_labels[i]])

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Clusters of Documents')
        plt.show()
