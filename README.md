# poly2SOP
Transformer takes a polynomial, expresses it as sum of powers.  Implemented with pytorch.

## Introduction


Tasks such as https://en.wikipedia.org/wiki/Sums_of_three_cubes, https://en.wikipedia.org/wiki/Taxicab_number and conjectures such as finding non - trivial integer solution of $a^5 + b^5 = c^5 + d^5$ requires deeper understanding of polynomial's characteristics. 

For example, if one finds non - trivial polynomial with no odd degree that is sum of 2 quintics ($ P(x^2) = p(x)^5 + q(x)^5$), Than by clever substitution :
$$
P(x^2) = P((-x)^2) = p(x)^5 + q(x)^5 = p(-x)^5 + q(-x)^5
$$

<p align="center" width="100%">
    <img src="https://latex.codecogs.com/svg.latex?P(x^2)=P((-x)^2)=p(x)^5+q(x)^5=p(-x)^5+q(-x)^5" title="P(x^2)=P((-x)^2)=p(x)^5+q(x)^5=p(-x)^5+q(-x)^5" />
</p>

This approach was actually first used by Ramanujan, in his set of parameterized solutions of $a^3 + b^3 = c^3 +d^3$.

Inspired from https://arxiv.org/pdf/1912.01412.pdf, I wondered if deep NNs are capable of such natural polynomial manipulations.

Here, we implement seq2seq model, using default pytorch's TransformerEncoder, TransformerDecoder module. Datasets were created with SymPy.

```python
from SOP_model import poly2SOP

chars = list("0987654321-+*()^xy")
n_vocab = len(chars) + 2 #One for paddings, one for init token.
device = torch.device("cuda:0")

model = poly2SOP(
    d_model = 512,
    n_head = 8,
    num_layers = 6,
    n_vocab = n_vocab, 
    max_len = max_len, 
    chars = chars,
    device = device
)
```



Simple use case with dataset I've created can be found in the repository too!

```python
...
chars = list("0987654321-+*()^xy")
n_vocab = len(chars) + 2

model = poly2SOP(
    d_model = 512,
    n_head = 8,
    num_layers = 6,
    n_vocab = n_vocab, 
    max_len = max_len, 
    chars = chars,
    device = device
)

opt = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-10)
dataset = eq_dataset(max_len = max_len, chars = chars)
dl = DataLoader(dataset, shuffle= True, batch_size= batch_size,  drop_last= True, num_workers = 3)
criterion = nn.CrossEntropyLoss()
model.to(device)

for epoch in range(1, epochs + 1):
    pbar = tqdm(dl)
    tot_loss = 0
    cnt = 0
    for (x, yin, yout) in pbar:

        x = x.to(device)
        yin = torch.cat([torch.ones(batch_size, 1) * (n_vocab - 1), yin], dim = 1).long()
        yin = yin.to(device)
        yout = yout.to(device)
        y_pred = model(x, yin)

        loss = criterion(y_pred.view(-1, n_vocab - 1), yout.view(-1))
        model.zero_grad()
        loss.backward()
        opt.step()
        tot_loss += loss.item()
        cnt += 1
        pbar.set_description(f"current loss : {tot_loss/cnt:.5f}")

    eq = "2*y^4-2*y^3-y^2+1"
    ans = "(1-y^2)^2+(-y^2+y)^2"

    ral = model.toSOP(eq, gen_len = max_len - 1)
    print(f'Epoch {epoch} : Loss : {tot_loss/cnt :.5f}, Example : {ral[0]}')

```



If you want, you can create your own datasets. 



```python
x, y = symbols('x y')
pol = [x, y, 1, x*y, x*x, y*y]
n = 2
def random_function(cr = 2):
    f = 0
    for mo in pol:
        f = f + mo*ri(-cr, cr)
    return expand(f)

# Later on...
f1, f2 = random_function(), random_function()
f3 = f1**n + f2**n
f4 = expand(f3)
FILE_x.write(str(f4).replace(' ', '').replace('**', '^') + '\n')
FILE_y.write(str(f3).replace(' ', '').replace('**', '^') + '\n')

```

