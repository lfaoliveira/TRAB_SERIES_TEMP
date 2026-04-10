
# Instalação e Execução
## Criação de ambiente local (IMPORTANTE!)
```bash
python -m venv ./.venv
```

## Instalar UV (não tem problema instalar no Python global)

```bash
pip install uv
```

## Instalar dependências

```bash
uv sync
```

## Se UV não for encontrado

#### Caso `uv` não seja reconhecido como programa, tente executar através do módulo Python:

```bash
python -m uv sync
```

#### Alternativamente, use o caminho completo para o executável do `uv` instalado no seu ambiente virtual:

```bash
./.venv/Scripts/uv sync  # Windows
./.venv/bin/uv sync      # Linux/Mac
```

## Rodar Notebook

```bash
jupyter notebook
```

Ou, se usar Jupyter manualmente no VS Code, apenas selecionar o ambiente da pasta clonada. 

```bash
uv run jupyter notebook
```
