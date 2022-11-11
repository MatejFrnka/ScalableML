import os
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(
        ["hopsworks", "seaborn", "joblib", "scikit-learn"])


    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("jim-hopsworks-ai"))
    def f():
        g()


def g():
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login()
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()

    # The feature view is the input set of features for your model. The features can come from different feature groups.
    # You can select features from different feature groups and join them together to create a feature view
    try:
        feature_view = fs.get_feature_view(name="titanic", version=1)
    except:
        titanic_fg = fs.get_feature_group(name="titanic", version=1)
        query = titanic_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic",
                                              version=1,
                                              description="Read from titanic dataset",
                                              labels=["survived"],
                                              query=query)

    X, y = feature_view.training_data()
    X = X.drop(['id'], axis=1)

    # Train our model with the Scikit-learn K-nearest-neighbors algorithm using our features (X_train) and labels (y_train)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y.values.ravel())

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()

    # The contents of the 'tmp_titanic_model' directory will be saved to the model registry. Create the dir, first.
    model_dir = "tmp_titanic_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/titanic_model.pkl")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X)
    output_schema = Schema(y)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    titanic_model = mr.python.create_model(
        name="titanic",
        model_schema=model_schema,
        description="Titanic survivor Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    titanic_model.save(model_dir)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
