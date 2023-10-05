from transformers import AutoModel

smiles = ["CCCC", "C#C/C(C)=C(/[CH2])C", "NNNNN"]

model = AutoModel.from_pretrained("Huhujingjing/custom-gcn", trust_remote_code=True)
