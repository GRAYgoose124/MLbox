import logging
import itertools
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from qiskit import Aer, QuantumCircuit, transpile, assemble, execute
from pathlib import Path


log = logging.getLogger(__name__)


class QcDSL:
    aer = Aer.get_backend("aer_simulator")

    def __init__(self, dsl, C):
        self.C = C
        self.dsl = dsl

    def new_qc_from_dsl(self, dsl, argvals):
        qc = QuantumCircuit(self.C["in"] - 1)  # -1 for circuit identifier

        for line in dsl.splitlines():
            line = line.strip()
            if line:
                tokens = line.split(",")
                method = tokens[0].strip()
                args = []
                for arg in tokens[1:]:
                    arg = arg.strip()
                    if arg.isnumeric():
                        if "." in arg:
                            args.append(float(arg))
                        else:
                            args.append(int(arg))
                    else:
                        if arg in argvals:
                            args.append(argvals[arg])
                        else:
                            args.append(arg)
                getattr(qc, method)(*args)

        return qc

    def get_quantum_output(self, argvals, n_shots=2048):
        log.debug(f"{argvals=}")
        qc = self.new_qc_from_dsl(self.dsl, argvals=argvals)

        t_qc = transpile(qc, QcDSL.aer)
        result = QcDSL.aer.run(t_qc, shots=n_shots).result()

        counts = result.get_counts(qc)
        probs = [
            counts.get(bin_str, 0) / n_shots
            for bin_str in [str(bin(i))[2:].zfill(2) for i in range(self.C["out"])]
        ]

        return probs


class QcNet(nn.Module):
    def __init__(self, C):
        super(QcNet, self).__init__()
        self.fc1 = nn.Linear(C["in"], 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, C["out"])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return torch.softmax(self.fc5(x), dim=1)


class QcNetTrainer:
    def __init__(self, C):
        # calculate input size from features
        C["in"] = C["in"] if "in" in C else len(C["features"])
        C["in"] += 1  # add circuit as it's own feature
        self.C = C
        self.model = QcNet(self.C)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.1,
        )
        self.criterion = nn.MSELoss()

        self._qcircuits = []

    @property
    def circuits(self):
        return self._qcircuits

    def add_dsl(self, dsl):
        self._qcircuits.append(QcDSL(dsl, self.C))

    def train(self, inputs, labels, epochs=1000):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def infer(self, circuit, **kwargs):
        # Add the circuit identifier to the args list
        args = [circuit]
        for line in self.circuits[circuit].dsl.splitlines():
            line = line.strip()
            if line:
                tokens = line.split(",")
                for token in tokens[1:]:
                    token = token.strip()
                    if token in kwargs:
                        args.append(kwargs[token])

        # infer
        with torch.no_grad():
            prediction = self.model(torch.tensor([args], dtype=torch.float32))
            log.info(
                f"For {args=}, predicted probabilities: {prediction.numpy()[0]}\n"
                f"Actual probabilities: {self.circuits[circuit].get_quantum_output(kwargs)}\n"
            )

    def gen_data_from_linspace(self, circuit, linspaces):
        """
        Generate inputs and labels using multiple linspaces and a given mapping.

        Args:
        - circuit: The circuit for which the data is being generated.
        - linspaces: A list of linspaces. Each linspace corresponds to an input.
        - mapping: A function that maps values from linspaces to desired argument values.

        Returns:
        - inputs: Torch tensor of input values.
        - labels: Torch tensor of labels generated from the circuit.
        """

        data = []
        inputs_list = []

        # Using itertools.product to generate all combinations of values from linspaces
        for values in itertools.product(*linspaces):
            feature_mapping = {f: v for (f, v) in zip(self.C["features"], values)}

            data.append(
                self.circuits[circuit].get_quantum_output(argvals=feature_mapping)
            )
            inputs_list.append([circuit, *values])
            # inputs_list.append([*values])
            log.debug(f"{inputs_list[-1]} {data[-1]}")

        inputs = torch.tensor(inputs_list, dtype=torch.float32)
        labels = torch.tensor(data, dtype=torch.float32)

        return inputs, labels

    def gen_and_train_over_all_circuits(self, linspaces, epochs=2500):
        """
        Generate data for all circuits and train the model over all circuits.

        Args:
        - linspaces: A list of linspaces. Each linspace corresponds to an input.
        - mapping: A function that maps values from linspaces to desired argument values.
        - epochs: Number of epochs to train for.
        """

        inputs_list = []

        for circuit in range(len(self.circuits)):
            log.info(f"Generating training data over circuit {circuit}")
            inputs, labels = self.gen_data_from_linspace(circuit, linspaces)
            # TODO: add circuit as it's own feature so network can start multiplexing it's predictions
            inputs_list.append((inputs, labels))

        # interlace inputs and labels
        inputs = torch.cat([i[0] for i in inputs_list])
        labels = torch.cat([i[1] for i in inputs_list])

        # train
        self.train(inputs, labels, epochs=epochs)

    def infer_input_over_all_circuits(self, **inputs):
        """
        Infer the output for a given input over all circuits.
        """

        for circuit in range(len(self.circuits)):
            log.info(f"Predicting output for circuit {circuit}")
            self.infer(circuit, **inputs)


def main():
    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("stevedore").setLevel(logging.WARNING)

    # init
    T = QcNetTrainer({"features": ["theta1", "theta2"], "out": 4})
    T.add_dsl(
        """
    ry, theta1, 0
    ry, theta2, 1
    cx, 0, 1
    measure_all
    """
    )
    T.add_dsl(
        """
    h, 0
    ry, theta1, 1
    cx, 0, 1
    ry, theta2, 0
    cx, 1, 0
    measure_all
    """
    )

    if Path("model.pt").exists() and input("load?").lower() == "y":
        T.model.load_state_dict(torch.load("model.pt"))
    log.info(T.model)

    T.gen_and_train_over_all_circuits(
        [np.linspace(0, 2 * np.pi), np.linspace(0, 2 * np.pi)],
        epochs=5000,
    )

    torch.save(T.model.state_dict(), "model.pt")

    T.infer_input_over_all_circuits(theta1=np.pi / 4, theta2=np.pi / 3)


if __name__ == "__main__":
    main()
