class Pipeline:
    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline needs at least one step.")
        self.steps = steps

    def _transform_steps(self, X):
        data = X

        for _, step in self.steps[:-1]:
            data = step.transform(data)

        return data

    def fit(self, X, y):
        data = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"{name} must implement transform()")

            if hasattr(step, "fit_transform"):
                data = step.fit_transform(data)
            else:
                step.fit(data)
                data = step.transform(data)

        model_name, model = self.steps[-1]

        if not hasattr(model, "predict"):
            raise TypeError(f"{model_name} must implement predict()")

        model.fit(data, y)
        return self

    def partial_fit(self, X, y):
        data = X

        for name, step in self.steps[:-1]:
            if hasattr(step, "partial_fit"):
                step.partial_fit(data)
            elif hasattr(step, "fit"):
                step.fit(data)
            else:
                raise TypeError(f"{name} must implement fit() or partial_fit()")

            if not hasattr(step, "transform"):
                raise TypeError(f"{name} must implement transform()")

            data = step.transform(data)

        model_name, model = self.steps[-1]

        if hasattr(model, "partial_fit"):
            model.partial_fit(data, y)
        elif hasattr(model, "fit"):
            model.fit(data, y)
        else:
            raise TypeError(f"{model_name} must implement fit() or partial_fit()")

        return self

    def predict(self, X):
        data = self._transform_steps(X)
        return self.steps[-1][1].predict(data)
