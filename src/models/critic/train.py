from .model import Critic
import tensorflow_datasets as tfds

train_ds, test_ds = tfds.load(
    "CoinrunC51", split=["train[:75%]", "train[75%:100%]"], as_supervised=True
)

model = Critic()

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

model.fit(train_ds)
print(model.summary())
