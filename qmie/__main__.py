import logging
import itertools
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from qiskit import Aer, QuantumCircuit, transpile, assemble, execute
from qiskit.circuit.random import random_circuit
from pathlib import Path


log = logging.getLogger(__name__)


class QcDSL:
    aer = Aer.get_backend("aer_simulator")

    def __init__(self, dsl, C):
        self.C = C
        self.dsl = dsl

    def new_qc_from_dsl(self, dsl, argvals):
        qc = QuantumCircuit(self.C["in"])

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

                if "_in" in args:
                    args[args.index("_in")] = self.C["in"]
                if "_out" in args:
                    args[args.index("_out")] = self.C["out"]

                if method == "random_circuit":
                    qc = random_circuit(self.C["in"])  # measure=True)
                else:
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
        self.fc1 = nn.Linear(
            C["in"] + 1, 32
        )  # +1 for circuit identifier (model multi-circuit support)
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


def file_cache(func):
    """Memoize to/from file based on args using UUID for filename."""
    import hashlib
    import uuid

    def wrapper(*args, **kwargs):
        arg_repr = [str(arg) for arg in args] + [
            f"{key}_{value}" for key, value in kwargs.items()
        ]
        arg_repr = "_".join(arg_repr[1:])
        hash_str = hashlib.md5((func.__name__ + "_" + arg_repr).encode()).hexdigest()
        filename = f"{func.__name__}_{hash_str}.pt"

        if Path(filename).exists():
            return torch.load(filename)
        else:
            result = func(*args, **kwargs)
            torch.save(result, filename)
            return result

    return wrapper


class QcNetTrainer:
    def __init__(self, C):
        # calculate input size from features
        C["in"] = C["in"] if "in" in C else len(C["features"])
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

    def get_args(self, circuit, **kwargs):
        args = []
        for line in self.circuits[circuit].dsl.splitlines():
            line = line.strip()
            if line:
                tokens = line.split(",")
                for token in tokens[1:]:
                    token = token.strip()
                    if token in kwargs:
                        args.append(kwargs[token])

        return args

    def infer(self, circuit, **kwargs):
        # Add the circuit identifier to the args list
        args = [circuit]
        args.extend(self.get_args(circuit, **kwargs))

        # infer
        with torch.no_grad():
            prediction = self.model(torch.tensor([args], dtype=torch.float32))
            log.info(
                f"For {args=}, predicted probabilities: {prediction.numpy()[0]}\n"
                f"Actual probabilities: {self.circuits[circuit].get_quantum_output(kwargs)}\n"
            )

    # @file_cache
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
            inputs_list.append((inputs, labels))

        num_samples_per_circuit = len(inputs_list[0][0])
        num_circuits = len(self.circuits)

        interlaced_inputs = []
        interlaced_labels = []

        for i in range(num_samples_per_circuit):
            for j in range(num_circuits):
                interlaced_inputs.append(inputs_list[j][0][i])
                interlaced_labels.append(inputs_list[j][1][i])

        inputs = torch.stack(interlaced_inputs)
        labels = torch.stack(interlaced_labels)

        self.train(inputs, labels, epochs=epochs)

    def infer_input_over_all_circuits(self, **inputs):
        """
        Infer the output for a given input over all circuits.
        """

        for circuit in range(len(self.circuits)):
            log.info(f"Predicting output for circuit {circuit}")
            self.infer(circuit, **inputs)


class QcTrainerManager:
    def __init__(self, feature_sets, outs):
        self.trainers = [
            QcNetTrainer({"features": fs, "out": out})
            for fs, out in zip(feature_sets, outs)
        ]

    def add_dsl_group(self, dsl_group, trainer_idx):
        for dsl in dsl_group:
            self.trainers[trainer_idx].add_dsl(dsl)

    def train_all(self, linspaces_list, epochs):
        for trainer, linspaces in zip(self.trainers, linspaces_list):
            trainer.gen_and_train_over_all_circuits(linspaces, epochs)

    def infer_all(self, **inputs):
        for trainer_idx, trainer in enumerate(self.trainers):
            for circuit_idx in range(len(trainer.circuits)):
                log.info(f"Trainer {trainer_idx}, Circuit {circuit_idx}")
                trainer.infer(circuit_idx, **inputs)

    def save_models(self):
        for idx, trainer in enumerate(self.trainers):
            model_path = Path(__file__) / f"model_{idx}.pt"
            torch.save(trainer.model.state_dict(), model_path)

    def load_models(self):
        for idx, trainer in enumerate(self.trainers):
            model_path = Path(__file__) / f"model_{idx}.pt"
            if Path(model_path).exists():
                trainer.model.load_state_dict(torch.load(model_path))


def gen_default_feature_set_linspace(n_features, out):
    feature_sets = [f"theta{n}" for n in range(n_features)]
    linspaces = [np.linspace(0, 2 * np.pi) for _ in range(n_features)]

    return feature_sets, linspaces, out


qubits1 = []
qubits2 = [
    """
ry, theta1, 0
ry, theta2, 1
cx, 0, 1
measure_all
""",
    """
h, 0
ry, theta1, 1
cx, 0, 1
ry, theta2, 0
cx, 1, 0
measure_all
""",
    #     """
    # random_circuit, _in, _out
    # measure_all
    # """,
]
qubits3 = []


# def main():
#     logging.basicConfig(level=logging.INFO, format="%(message)s")
#     logging.getLogger("qiskit").setLevel(logging.WARNING)
#     logging.getLogger("stevedore").setLevel(logging.WARNING)

#     feature_sets, linspaces_list, outs = gen_default_feature_set_linspace(2, 4)
#     feature_sets, linspaces_list, outs = [feature_sets], [linspaces_list], [outs]

#     manager = QcTrainerManager(feature_sets, outs)

#     # Add DSL groups to corresponding trainers
#     manager.add_dsl_group(qubits2, 0)

#     # Load models (optional, if needed)
#     # manager.load_models()

#     manager.train_all(linspaces_list, epochs=5000)

#     # Save models
#     manager.save_models()

#     # Run inference
#     manager.infer_all(theta1=np.pi / 4, theta2=np.pi / 3)


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
