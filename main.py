import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load and preprocess the training data
train_data = pd.read_csv('train.csv')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].fillna('S')

# Fit the encoder on the training data so it can be used to transform both training and test data
embarked_label_encoder = LabelEncoder()
embarked_label_encoder.fit(train_data['Embarked'])
train_data['Embarked'] = embarked_label_encoder.transform(train_data['Embarked'])

# Select features and target variable for the training data
X_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = train_data['Survived']

# Initialize and train the decision tree classifier on the training data split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_split, y_train_split)

# Optionally evaluate the model using part of your training data
y_pred_split = decision_tree.predict(X_test_split)
accuracy = accuracy_score(y_test_split, y_pred_split)
print(f'Accuracy of the decision tree classifier on split test set: {accuracy:.2f}')

# Preprocess the test data
test_data = pd.read_csv('test.csv')
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data['Embarked'] = embarked_label_encoder.transform(test_data['Embarked'])

# Select the same features for the test data
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Predict using the trained decision tree classifier
predictions = decision_tree.predict(X_test)

# Generate a submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)

# Visualization of the Decision Tree based on the training data.
plt.figure(figsize=(20, 10))  # set plot size (width, height)
tree.plot_tree(decision_tree, feature_names=X_train.columns, class_names=['Not Survived', 'Survived'], filled=True)

# Save the plot to a file
plt.savefig('decision_tree.png', format='png', dpi=1080, bbox_inches='tight')

# Finnish
print("Prediction file 'submission.csv' created, and decision tree visualized.")