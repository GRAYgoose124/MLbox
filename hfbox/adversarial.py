import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
pipe = pipeline("text-generation", model="distilgpt2", device=0)


def get_gpt2_output(prompt):
    text = pipe(prompt, min_new_tokens=99, max_new_tokens=100)[0]["generated_text"]
    return text


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 100)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def adversarial_training(generator, predictor):
    opt_gen = optim.Adam(generator.parameters(), lr=0.001)
    opt_pred = optim.Adam(predictor.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    # Changed this to have a single element, matching the generated data batch size
    actual_labels = torch.randint(0, 2, (1,))

    for epoch in range(10):  # Reduced the epochs to 10 for illustration
        prompt = "Once"

        with torch.no_grad():
            gpt2_output = get_gpt2_output(prompt)
            gpt2_tokens = (
                torch.tensor(tokenizer.encode(gpt2_output)).unsqueeze(0).float()
            ).clone()

        noise = torch.randn(1, 100)
        generated_data = generator(noise).clone()

        if generated_data.shape[1] != gpt2_tokens.shape[1]:
            min_len = min(generated_data.shape[1], gpt2_tokens.shape[1])
            generated_data = generated_data[:, :min_len]
            gpt2_tokens = gpt2_tokens[:, :min_len]

        fixed_size = 100
        if generated_data.shape[1] < fixed_size:
            padding = torch.zeros(
                (generated_data.shape[0], fixed_size - generated_data.shape[1])
            )
            generated_data = torch.cat([generated_data, padding], dim=1)

            gpt2_padding = torch.zeros(
                (gpt2_tokens.shape[0], fixed_size - gpt2_tokens.shape[1])
            )
            gpt2_tokens = torch.cat([gpt2_tokens, gpt2_padding], dim=1)

        # IMPORTANT: Truncate or pad `generated_data` and `gpt2_tokens` to the same size here

        loss_gen_to_gpt2 = nn.MSELoss()(generated_data, gpt2_tokens)
        opt_gen.zero_grad()
        loss_gen_to_gpt2.backward(retain_graph=True)
        opt_gen.step()

        generated_data.detach()

        prediction = predictor(generated_data)
        loss_pred = criterion(prediction, actual_labels)
        opt_pred.zero_grad()
        loss_pred.backward(retain_graph=True)
        opt_pred.step()

        generated_data.detach()

        generated_data = generator(noise)
        prediction = predictor(generated_data)
        loss_gen = -criterion(prediction, actual_labels)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if epoch % 1 == 0:
            P = loss_pred.item()
            G = -loss_gen.item()

            print(f"E:{epoch+1}\t{P:.3f}{G:.3f} P-G:{P-G:.3f}")


def main():
    torch.autograd.set_detect_anomaly(True)

    if input("Load models? (y/N): ").lower() == "y":
        generator = Generator()
        generator.load_state_dict(torch.load("generator.pt"))
        generator.eval()
        predictor = Predictor()
        predictor.load_state_dict(torch.load("predictor.pt"))
        predictor.eval()
    else:
        generator = Generator()
        predictor = Predictor()

    try:
        adversarial_training(generator, predictor)
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(generator.state_dict(), "generator.pt")
        torch.save(predictor.state_dict(), "predictor.pt")


if __name__ == "__main__":
    main()
