import torch.nn as nn
import torch.nn.functional as F
import torch
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from typing import Dict, List, Optional, Tuple

class ClassifierRegressor(nn.Module):
    def __init__(self, input_dim=1, num_classes=1):
        super(ClassifierRegressor, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.classifier = nn.Linear(64, self.num_classes )  # Classification head
        self.regressor = nn.Linear(64, 1)     # Regression head



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        class_output = self.classifier(x)
        reg_output = self.regressor(x)
        return class_output, reg_output
    
class ClassifierRejector(nn.Module):
    def __init__(self, input_dim=1, num_experts = 1):
        super(ClassifierRejector, self).__init__()
        self.num_classes = num_experts + 1
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.classifier = nn.Linear(64, self.num_classes )  # Classification head

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        class_output = self.classifier(x)
        return class_output


class MLP(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            pretrained_emb: str = None,
            embedding_dim: int = 256,
            hidden_dim: int = 256,
            n_layers: int = 6,
            activation: str = "relu",
            **kwargs,
    ):
        super(MLP, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature MLP layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "MLP only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [1, 2]):
                raise ValueError(
                    "MLP only supports 1-dim or 2-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [1, 2, 3]
            ):
                raise ValueError(
                    "MLP only supports 1-dim, 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function {activation}")

        self.mlp = nn.ModuleDict()
        for feature_key in feature_keys:
            Modules = []
            Modules.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            for _ in range(self.n_layers - 1):
                Modules.append(self.activation)
                Modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.mlp[feature_key] = nn.Sequential(*Modules)

        output_size = self.get_output_size(self.label_tokenizer)
        self.classifier = nn.Sequential(
            nn.Linear(len(self.feature_keys) * self.hidden_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Dropout layer to prevent overfitting

            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Another Dropout layer

            nn.Linear(64, output_size)  # Output layer
        )
        self.regressor = nn.Sequential(
            nn.Linear(len(self.feature_keys) * self.hidden_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Dropout layer to prevent overfitting

            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Another Dropout layer

            nn.Linear(64, 1)  # Output layer
        )

    @staticmethod
    def mean_pooling(x, mask):
        """Mean pooling over the middle dimension of the tensor.
        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)
        Returns:
            x: tensor of shape (batch_size, embedding_dim)
        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> mean_pooling(x).shape
            [128, 32]
        """
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    @staticmethod
    def sum_pooling(x):
        """Mean pooling over the middle dimension of the tensor.
        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)
        Returns:
            x: tensor of shape (batch_size, embedding_dim)
        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> sum_pooling(x).shape
            [128, 32]
        """
        return x.sum(dim=1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        data = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                data.append(x)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x != 0, dim=2)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x != 0, dim=2)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 3: [1.5, 2.0, 0.0]
            elif (dim_ == 1) and (type_ in [float, int]):
                # (patient, values)
                x = torch.tensor(
                    kwargs[feature_key], dtype=torch.float, device=self.device
                )
                # (patient, embedding_dim)
                x = self.linear_layers[feature_key](x)

            # for case 4: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 5: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            else:
                raise NotImplementedError

            if self.pretrained_emb != None:
                x = self.linear_layers[feature_key](x)

            x = self.mlp[feature_key](x)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        classifier_fc = self.classifier(patient_emb)
        regressor_fc = self.regressor(patient_emb)
        # obtain y_true, loss, y_prob
        label_class = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        label_reg = torch.tensor(kwargs['label_length_of_stay'], dtype=torch.float, device=self.device)
        results = {
            'classifier_fc': classifier_fc,
            'regressor_fc': regressor_fc,
            'label_class': label_class,
            'label_reg': label_reg
        }
        return results, data


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Dict


class TransformerModel(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            pretrained_emb: str = None,
            embedding_dim: int = 256,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            **kwargs,
    ):
        super(TransformerModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Embeddings
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, embedding_dim))  # Adjust 1000 to max sequence length

        # add feature MLP layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Transformer only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Classifier and Regressor
        output_size = self.get_output_size(self.label_tokenizer)
        self.classifier = nn.Sequential(
            nn.Linear(dim_feedforward, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        self.regressor = nn.Sequential(
            nn.Linear(dim_feedforward, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        # Process each feature
        patient_emb = []
        data = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            if (dim_ == 2) and (type_ == str):
                # Tokenize and embed
                x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                data.append(x)
                x = self.embeddings[feature_key](x)

            elif (dim_ == 1) and (type_ in [float, int]):
                x = torch.tensor(kwargs[feature_key], dtype=torch.float, device=self.device)
                x = self.linear_layers[feature_key](x)
            else:
                raise NotImplementedError

            # Add positional encoding
            x += self.positional_encoding[:, :x.size(1), :]

            # Transformer encoding
            x = self.transformer_encoder(x)

            # Pooling: Take the mean of the output embeddings
            x = x.mean(dim=1)
            patient_emb.append(x)

        # Concatenate feature embeddings
        patient_emb = torch.cat(patient_emb, dim=1)

        # Classification and regression
        classifier_fc = self.classifier(patient_emb)
        regressor_fc = self.regressor(patient_emb)

        label_class = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        label_reg = torch.tensor(kwargs['label_length_of_stay'], dtype=torch.float, device=self.device)
        results = {
            'classifier_fc': classifier_fc,
            'regressor_fc': regressor_fc,
            'label_class': label_class,
            'label_reg': label_reg
        }
        return results, data


class ClassifierRejector(BaseModel):

    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            pretrained_emb: str = None,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            n_layers: int = 4,
            activation: str = "relu",
            num_output: int = 1,
            **kwargs,
    ):
        super(ClassifierRejector, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature MLP layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "MLP only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [1, 2]):
                raise ValueError(
                    "MLP only supports 1-dim or 2-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [1, 2, 3]
            ):
                raise ValueError(
                    "MLP only supports 1-dim, 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function {activation}")

        self.mlp = nn.ModuleDict()
        for feature_key in feature_keys:
            Modules = []
            Modules.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            for _ in range(self.n_layers - 1):
                Modules.append(self.activation)
                Modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.mlp[feature_key] = nn.Sequential(*Modules)

        self.rejector = nn.Sequential(
            nn.Linear(len(self.feature_keys) * self.hidden_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0),  # Optional: Dropout layer to prevent overfitting

            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0),  # Optional: Another Dropout layer

            nn.Linear(64, num_output)  # Output layer
        )


    @staticmethod
    def mean_pooling(x, mask):
        """Mean pooling over the middle dimension of the tensor.
        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)
        Returns:
            x: tensor of shape (batch_size, embedding_dim)
        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> mean_pooling(x).shape
            [128, 32]
        """
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    @staticmethod
    def sum_pooling(x):
        """Mean pooling over the middle dimension of the tensor.
        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)
        Returns:
            x: tensor of shape (batch_size, embedding_dim)
        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> sum_pooling(x).shape
            [128, 32]
        """
        return x.sum(dim=1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x != 0, dim=2)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x != 0, dim=2)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 3: [1.5, 2.0, 0.0]
            elif (dim_ == 1) and (type_ in [float, int]):
                # (patient, values)
                x = torch.tensor(
                    kwargs[feature_key], dtype=torch.float, device=self.device
                )
                # (patient, embedding_dim)
                x = self.linear_layers[feature_key](x)

            # for case 4: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 5: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            else:
                raise NotImplementedError

            if self.pretrained_emb != None:
                x = self.linear_layers[feature_key](x)

            x = self.mlp[feature_key](x)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        rejector_fc = self.rejector(patient_emb)
        # obtain y_true, loss, y_prob
        label_class = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        label_reg = torch.tensor(kwargs['label_length_of_stay'], dtype=torch.float, device=self.device)
        results = {
            'rejector_fc': rejector_fc,
            'label_class': label_class,
            'label_reg': label_reg
        }
        return results





class Custom(BaseModel):
    """Multi-layer perceptron model.

    This model applies a separate MLP layer for each feature, and then concatenates
    the final hidden states of each MLP layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        We use separate MLP layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the rnn model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector; we use mean/sum pooling and then MLP
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we first use the embedding table to encode each code into a vector
                and then use mean/sum pooling to get one vector for each sample; we then
                use MLP layers
            - case 3. [1.5, 2.0, 0.0]
                - we run MLP directly
            - case 4. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - This case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we use mean/sum pooling
                within each outer bracket and use MLP, similar to case 1 after embedding table
            - case 5. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - This case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we use mean/sum pooling
                within each outer bracket and use MLP, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        n_layers: the number of layers. Default is 2.
        activation: the activation function. Default is "relu".
        **kwargs: other parameters for the RNN layer.

    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [1.0, 2.0, 3.5, 4],
        ...             "label": 0,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [5.0, 2.0, 3.5, 4],
        ...             "label": 1,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import MLP
        >>> model = MLP(
        ...         dataset=dataset,
        ...         feature_keys=["conditions", "procedures"],
        ...         label_key="label",
        ...         mode="binary",
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6659, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.5680],
                            [0.5352]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                            [0.]]),
            'logit': tensor([[0.2736],
                            [0.1411]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            pretrained_emb: str = None,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            n_layers: int = 4,
            activation: str = "relu",
            **kwargs,
    ):
        super(Custom, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature MLP layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "MLP only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [1, 2]):
                raise ValueError(
                    "MLP only supports 1-dim or 2-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [1, 2, 3]
            ):
                raise ValueError(
                    "MLP only supports 1-dim, 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function {activation}")

        self.mlp = nn.ModuleDict()
        for feature_key in feature_keys:
            Modules = []
            Modules.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            for _ in range(self.n_layers - 1):
                Modules.append(self.activation)
                Modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.mlp[feature_key] = nn.Sequential(*Modules)

        output_size = self.get_output_size(self.label_tokenizer)
        self.classifier = nn.Sequential(
            nn.Linear(len(self.feature_keys) * self.hidden_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Dropout layer to prevent overfitting

            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Another Dropout layer

            nn.Linear(64, output_size)  # Output layer
        )
        self.regressor = nn.Sequential(
            nn.Linear(len(self.feature_keys) * self.hidden_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Dropout layer to prevent overfitting

            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.1),  # Optional: Another Dropout layer

            nn.Linear(64, 1)  # Output layer
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        store = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                store.append(torch.tensor(x, dtype=torch.long, device=self.device))


        return store
