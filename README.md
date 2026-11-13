# Documento Fácil

> **IA para organização inteligente de documentos de interesse público**

Aplicação web com Gemini Grounding para pesquisar informação pública na web, resumir a resposta e apresentar links das fontes consultadas.

---

## Publicar no Netlify

O site publicado usa `public/` e a função Node.js em `netlify/functions/answer.mjs`. A busca web e a geração acontecem no servidor Netlify; a chave Gemini nunca vai para o navegador.

1. No Netlify, importe o repositório `Luizfelipe075/documento-facil`.
2. O Netlify lê `netlify.toml`: comando de build `npm run build`, pasta publicada `public` e funções em `netlify/functions`.
3. Em **Project configuration → Environment variables**, adicione `GEMINI_API_KEY` com a chave do Google AI Studio.
4. Publique/republique o site. A página chama `/api/answer`; a função pesquisa a web com Gemini e retorna a resposta e as fontes.

Não coloque a chave no código, no navegador ou em arquivos enviados ao GitHub. Configure-a apenas nas variáveis de ambiente do Netlify.

### Executar localmente

Requer Node.js 24 (definido em `.nvmrc`) e npm:

```powershell
npm install
$env:GEMINI_API_KEY = "sua_chave_aqui"
npm run dev
```

O script baixa o Netlify CLI quando necessário e abre o site e as funções em `http://localhost:8888`. Os passos aparecem na área **Atividade do fluxo** da página e os registros do servidor aparecem no terminal. No site publicado, os logs do servidor ficam no painel de funções do Netlify.

A função limita cada endereço IP a 8 perguntas por minuto. Isso reduz abuso acidental do endpoint público; configure também limites de uso para a chave Gemini na sua conta Google.

### O que a IA responde

O Gemini faz uma busca web nova para cada pergunta. Ele pode responder sobre assuntos públicos que encontrar e citar; não conhece fatos privados nem deve ser tratado como fonte universal. Quando a busca não retorna fontes verificáveis suficientes, o app informa que não conseguiu confirmar a resposta. Para legislação e serviços públicos brasileiros, as instruções priorizam fontes oficiais.

O pipeline Python/Streamlit abaixo foi mantido para pesquisa e uso local. Ele não é o backend do site Netlify.

---

## Funcionalidades do pipeline Python local (legado)

- **Coleta automática** de legislação federal do Portal do Planalto
- **Extração de texto** de PDFs (PyMuPDF) e páginas HTML (BeautifulSoup)
- **Limpeza e normalização** preservando estrutura jurídica (artigos, incisos, alíneas)
- **Chunking** por tokens com sobreposição configurável
- **Embeddings multilíngues** com `intfloat/multilingual-e5-large` (ótimo para português)
- **Índice vetorial** persistente com ChromaDB
- **Busca semântica** por similaridade cosseno
- **Geração de respostas** baseadas exclusivamente nos documentos recuperados (Gemini ou OpenAI)
- **Interface Streamlit** com fontes clicáveis e aviso de IA
- **Avaliação** de recuperação (Recall@K, MRR) e qualidade de respostas, com **experimentos comparativos** de chunking

---

## Estrutura do projeto

```
documento-facil/
├── data/
│   ├── raw/pdf/              # PDFs coletados
│   ├── raw/html/             # HTMLs coletados
│   ├── processed/
│   │   ├── documents.jsonl   # Metadados dos documentos
│   │   └── chunks.jsonl      # Chunks processados
│   ├── questions/
│   │   ├── test_questions.json       # 20 perguntas de teste
│   │   ├── evidence_ldb.json         # evidências por pergunta (métricas por trecho)
│   │   └── document_aliases.json     # liga IDs das perguntas aos IDs reais do coletor
│   └── experiments/                  # saídas dos experimentos (gerado)
├── src/
│   ├── collectors/           # Coleta de documentos
│   ├── extraction/           # Extração de PDF e HTML
│   ├── processing/           # Limpeza e chunking
│   ├── embeddings/           # Geração de embeddings
│   ├── retrieval/            # Índice vetorial e busca
│   ├── generation/           # LLM e pipeline RAG
│   └── evaluation/           # Métricas e experimentos de avaliação
│       ├── retrieval.py          #   Recall@K simples
│       ├── generation.py         #   protocolo de avaliação manual das respostas
│       ├── metrics.py            #   métricas puras (Recall@K, MRR, limiar...)
│       └── experiments.py        #   TASK 17: experimentos comparativos
├── app/
│   └── streamlit_app.py      # Interface do usuário
├── public/                   # Site publicado pelo Netlify
├── netlify/functions/        # API serverless com Gemini e busca web
├── netlify.toml              # Configuração de build, rotas e funções
├── scripts/
│   ├── poc.py                # Prova de conceito (1 documento)
│   ├── run_pipeline.py       # Pipeline completo
│   └── run_experiments.py    # Experimentos de avaliação (TASK 17)
├── tests/                    # Testes unitários (rodam offline)
│   └── fixtures/             # documento sintético para testes
├── vectorstore/              # Índice ChromaDB (gerado automaticamente)
├── requirements.txt
└── .env.example
```

---

## Instalação do protótipo Python local (legado)

### Abrir no PyCharm (Windows)

1. Extraia o ZIP e abra a pasta `documento-facil` no PyCharm.
2. Em **Settings > Project > Python Interpreter**, crie um ambiente virtual com Python 3.11 ou superior.
3. No terminal do PyCharm, instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

4. Para salvar a chave, crie `.env` na pasta do projeto (ao lado de `requirements.txt`) e adicione:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

O `.env` serve apenas para executar o protótipo Python local. Para o site Netlify, use a variável `GEMINI_API_KEY` nas configurações privadas do projeto Netlify. Nunca publique a chave.

5. Para abrir a interface:

```powershell
python -m streamlit run app/streamlit_app.py
```

O navegador abre em `http://localhost:8501`. Para fazer perguntas, também é preciso criar o índice com documentos; veja **Fase 2 — Pipeline completo** abaixo. O primeiro uso baixa o modelo local de embeddings e precisa de conexão com a internet. As etapas da busca e geração aparecem no site e os detalhes técnicos no terminal que iniciou o Streamlit.

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd documento-facil
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha:

```env
# Provedor LLM: gemini (padrão) ou openai
LLM_PROVIDER=gemini
GOOGLE_API_KEY=sua_chave_aqui

# Ou OpenAI:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sua_chave_aqui
```

> **Nota:** Para obter uma chave do Google Gemini gratuitamente, acesse [Google AI Studio](https://aistudio.google.com/).
>
> **Atenção ao modelo:** os modelos `gemini-1.5-*` foram descontinuados pelo Google e retornam erro 404.
> O padrão agora é o alias `gemini-flash-latest` (variável `GEMINI_MODEL`). Se sua chave não aceitar,
> consulte os modelos disponíveis em <https://ai.google.dev/gemini-api/docs/models>. Se você já tem um
> `.env` antigo, atualize `GEMINI_MODEL` nele.

---

## Uso

### Fase 1 — Prova de conceito (1 documento)

```bash
# Modo demo: baixa a LDB do Planalto e faz uma pergunta
python scripts/poc.py --demo

# Com seu próprio arquivo:
python scripts/poc.py --file data/raw/html/minha_lei.html --question "O que esta lei estabelece?"
python scripts/poc.py --file data/raw/pdf/decreto.pdf --question "Qual é o objeto deste decreto?"
```

### Fase 2 — Pipeline completo

```bash
# Coleta + processa + indexa (50 documentos por padrão)
python scripts/run_pipeline.py

# Coletar apenas 10 documentos
python scripts/run_pipeline.py --limit 10

# Pular coleta (reprocessar docs já baixados)
python scripts/run_pipeline.py --skip-collect

# Reindexar com chunks já gerados
python scripts/run_pipeline.py --skip-extract --reset-index
```

### Fase 3 — Interface Streamlit local

```bash
streamlit run app/streamlit_app.py
```

Acesse em: http://localhost:8501

### Avaliação

```bash
# Calcular Recall@K
python -c "
from pathlib import Path
from src.evaluation.retrieval import evaluate_retrieval
result = evaluate_retrieval(Path('data/questions/test_questions.json'))
print(result['metrics'])
"

# Gerar respostas para avaliação manual
python -c "
from pathlib import Path
from src.evaluation.generation import run_evaluation
run_evaluation(
    Path('data/questions/test_questions.json'),
    Path('data/evaluation_results.json')
)
"
```

### Experimentos de avaliação (TASK 17)

Compara configurações de **chunking** (tamanho × sobreposição) medindo a qualidade da recuperação
(Recall@K, MRR) e, opcionalmente, das respostas geradas. Cada configuração usa um índice ChromaDB
**isolado e temporário** — o `vectorstore/` do app não é alterado.

```bash
# 1) Tenha documentos coletados (um dos dois):
python scripts/poc.py --demo
python scripts/run_pipeline.py

# 2) Confira se as evidências das perguntas existem no SEU corpus (recomendado, rápido):
python scripts/run_experiments.py --check-evidence

# 3) Rode os experimentos
python scripts/run_experiments.py              # grade padrão: 250/500/800 tokens x sobreposição 0/100
python scripts/run_experiments.py --quick      # 2 configurações, para um primeiro teste
python scripts/run_experiments.py --chunk-sizes 300 600 --overlaps 50 100 --k-values 1 3 5 10
python scripts/run_experiments.py --generation # + avalia as respostas do LLM (consome a API; precisa de .env)
```

O embedding com `multilingual-e5-large` em CPU é a etapa mais demorada: comece com `--quick`.

**Sem internet / sem baixar o modelo** (valida só o funcionamento, não a qualidade):

```bash
python scripts/run_experiments.py --demo-fixture              # corpus sintético + embedder offline
python scripts/run_experiments.py --embedder hashing          # seu corpus + baseline lexical offline
```

**Saídas** em `data/experiments/<data_hora>/`:

| Arquivo | Conteúdo |
|---------|----------|
| `report.md` | Relatório pronto para alimentar o relatório técnico (TASK 20) |
| `summary.csv` | Uma linha por configuração (abre direto no Excel) |
| `results.json` | Resultados completos, por pergunta |
| `generation_results.json` | Respostas e checagens automáticas (com `--generation`) |
| `generation_manual_eval.csv` | Planilha para a **avaliação manual** (correta, relevante, fundamentada, fonte correta, alucinação) |
| `run.log` | Log da execução |

**Como interpretar os resultados**

- **Recall por documento vs. por evidência.** Todas as perguntas esperam o mesmo documento (LDB). Se só ele
  estiver indexado, o Recall por documento é 100% em qualquer configuração e não diferencia nada. Por isso o
  relatório também mede por **evidência**: algum trecho do Top-K contém o texto que responde à pergunta?
  As evidências ficam em `data/questions/evidence_ldb.json` e são **heurísticas** — valide com `--check-evidence`.
- **Chunks maiores têm vantagem mecânica**: contêm mais texto, então acertam mais, mas enviam mais tokens ao LLM.
  Leia a coluna *Contexto* junto com o Recall.
- **Perguntas sem resposta** (q10, q11, q20): o relatório sugere um limiar de similaridade para o sistema se
  recusar a responder. Com 3 perguntas, é só um indicativo.
- **Checagens de geração são heurísticas** (detecção de recusa por frases, presença de fontes e aviso de IA).
  Não substituem a avaliação manual: use `generation_manual_eval.csv`.
- **Corpus com mais documentos** (via `run_pipeline.py`) torna o Recall por documento informativo. IDs gerados
  pelo coletor são ligados aos das perguntas por `data/questions/document_aliases.json`.

### Testes

```bash
pytest tests/ -v          # todos rodam offline (sem baixar modelos nem chamar APIs)
```

---

## Modelo de dados

### Documento

```json
{
  "document_id": "lei_9394_1996_a1b2c3d4",
  "title": "Lei no 9.394, de 20 de dezembro de 1996",
  "document_type": "lei",
  "organization": "Presidência da República",
  "publication_date": "20 de dezembro de 1996",
  "source_url": "https://www.planalto.gov.br/ccivil_03/leis/l9394.htm",
  "collected_at": "2026-10-09T17:00:00+00:00",
  "file_path": "data/raw/html/lei_9394_1996_a1b2c3d4.html",
  "content_type": "html"
}
```

### Chunk

```json
{
  "chunk_id": "lei_9394_1996_a1b2c3d4_chunk_0042",
  "document_id": "lei_9394_1996_a1b2c3d4",
  "text": "Art. 3º O ensino será ministrado com base nos seguintes princípios...",
  "chunk_index": 42,
  "source_url": "https://www.planalto.gov.br/ccivil_03/leis/l9394.htm",
  "title": "Lei no 9.394, de 20 de dezembro de 1996",
  "organization": "Presidência da República",
  "publication_date": "20 de dezembro de 1996",
  "token_count": 487
}
```

---

## Tecnologias

| Componente | Tecnologia | Justificativa |
|------------|-----------|---------------|
| Coleta HTTP | `requests` + `BeautifulSoup` | Simples, sem overhead de JS |
| Extração PDF | `PyMuPDF` | Rápido, mantém posição por página |
| Embeddings | `multilingual-e5-large` | Excelente para português, local e gratuito |
| Índice vetorial | `ChromaDB` | Persiste metadados e vetores juntos |
| LLM (padrão) | Google Gemini | Gratuito com limites generosos |
| Interface | `Streamlit` | Rápido para prototipagem Python |

---

## Princípio fundamental

```
DOCUMENTO OFICIAL  ≠  RESPOSTA DA IA
```

O sistema **nunca inventa informações**. Se os documentos recuperados não forem suficientes para responder, o sistema informa claramente que não encontrou informação suficiente. Toda resposta exibe:

- Título do documento fonte
- URL do documento oficial
- Aviso de conteúdo gerado por IA

---

## Aviso

> Este sistema é um projeto acadêmico de pesquisa. As respostas geradas por IA são baseadas em documentos públicos recuperados automaticamente e **não constituem orientação jurídica**. Consulte sempre o documento oficial e um profissional qualificado.

---

## Fontes de dados

| Portal | URL | Prioridade |
|--------|-----|-----------|
| Planalto — Legislação | https://www4.planalto.gov.br/legislacao | Alta |
| Câmara dos Deputados | https://www.camara.leg.br/legislacao/ | Alta |
| Senado Federal | https://www25.senado.leg.br/web/atividade/legislacao | Alta |
| Diário Oficial da União | https://www.in.gov.br/ | Alta |
| Portal da Transparência | https://portaldatransparencia.gov.br/ | Média |
