# FAQ

The FAQ assistant for UmbrellaLabs and UmbrellaNET. Visitors ask a question in plain language and get a clear answer based on the company's own documents and web pages.

License: [add a license, for example MIT]

UmbrellaLabs sells climate-positive AI automation to small businesses and enterprises, powered by UmbrellaNET's climate-positive managed cloud. This service answers the questions those visitors ask: what the services are, how pricing and ROI commitments work, what the climate claims mean, and how to get in touch.

The live service runs at https://faq-two-murex.vercel.app

> Items in [square brackets] are details I could not confirm from the files I was given (`vercel.json`, `requirements.txt` and the UmbrellaLabs home page). Replace them with the real values or delete them.

## Contents

- [How it works](#how-it-works)
- [How the assistant was built](#how-the-assistant-was-built)
- [Hosting on Vercel](#hosting-on-vercel)
- [Getting started](#getting-started)
- [Configuration reference](#configuration-reference)
- [Repository layout](#repository-layout)
- [Testing and verification](#testing-and-verification)
- [Adding or changing FAQ content](#adding-or-changing-faq-content)
- [Accuracy and data protection](#accuracy-and-data-protection)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## How it works

The service is a Flask app deployed on Vercel as a Python serverless function. It has no GPU and no database of its own. It calls Google's Gemini API to write answers, and reads its source material from documents and pages that you provide.

What happens to one question:

1. The visitor types a question on the website [or in the page served from `templates/`].
2. The browser sends the question to the API under `/api/`.
3. The app loads the FAQ source material: text pulled from PDF files, plus any web pages the content links to.
4. The question and the relevant source text go to Gemini with instructions to answer only from that material.
5. The answer is returned as JSON and shown to the visitor.
6. Links are returned in the answer.



## How the assistant was built

The main design decisions, and why:

### 1. Answers come from your documents, not from the model's memory

The company's facts live in PDFs and web pages, read with PyMuPDF. Gemini is instructed to answer from that text. This keeps answers about services, pricing and climate claims tied to what UmbrellaLabs has actually published.

### 2. A thin API that other sites can call

`flask-cors` allows the UmbrellaLabs website (a separate Next.js site) to call this API from the browser. Restrict the allowed origins to your own domains in production (see [Configuration](#configuration-reference)).

### 3. Serverless by default

`vercel.json` builds `api/index.py` with the Vercel Python runtime and sends every `/api/*` request to it. Nothing is stored on the function's disk between requests.

### 4. Safe defaults for company claims

UmbrellaLabs makes specific commitments on its site, such as ROI guarantees and carbon reduction figures. The assistant should repeat only what the source documents say, and send people to the contact form for anything it cannot answer. [Confirm the system prompt in `api/` says this.]

## Hosting on Vercel

1. Push this repository to GitHub.
2. In Vercel, choose Add New, Project, and import `UmbrellaNET/FAQ`.
3. Add the environment variables from the [configuration reference](#configuration-reference).
4. Deploy. Every push to `main` redeploys automatically.

Notes:

- `vercel.json` uses `"version": 2` with a `builds` entry for `api/index.py` and one route, `/api/(.*)`. Requests outside `/api/` are not sent to the function by this file.
- Redeploy after changing environment variables. A running deployment does not pick them up.
- Vercel functions have a maximum duration and package size set by your plan. Calls to Gemini can be slow, so keep source text short and set a request timeout.

## Getting started

### Prerequisites

- Python 3.10 or newer
- A Google AI Studio key for Gemini
- A GitHub account and a Vercel account for deployment

### 1. Clone and install

```bash
git clone https://github.com/UmbrellaNET/FAQ.git
cd FAQ
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root (`python-dotenv` loads it locally). Never commit it.

```bash
[GEMINI_API_KEY=your-gemini-key]
```

### 3. Run locally

```bash
[python3 api/index.py]
```

Then try the API:

```bash
curl -X POST http://localhost:[5000]/api/[ask] \
  -H "Content-Type: application/json" \
  -d '{"question":"What does UmbrellaLabs do?"}'
```

### 4. Deploy

Push to `main`, or use the Vercel CLI:

```bash
npm i -g vercel
vercel --prod
```

## Configuration reference

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| [GEMINI_API_KEY] | yes | | Gemini key used to write answers |
| [ALLOWED_ORIGINS] | no | | Comma separated sites allowed to call the API (CORS) |

Set these in Vercel under Project Settings, Environment Variables. Use the exact names your code reads.

## Repository layout

```
api/                 Flask app (entry point api/index.py) run by Vercel as a function
templates/           HTML templates for pages the app serves
info.txt             [what this file holds]
requirements.txt     Python dependencies
vercel.json          Vercel build and routing configuration
README.md            This file
```

Dependencies, and what each is for:

| Package | Use |
| --- | --- |
| Flask | Web framework |
| flask-cors | Lets the UmbrellaLabs site call the API from the browser |
| google-generativeai | Gemini client for writing answers |
| requests | Fetching web pages |
| python-dotenv | Loading `.env` for local development |
| PyMuPDF | Reading text from PDF files |
| urlextract | Finding URLs in text |

## Testing and verification

Before deploying a change:

1. Run the app locally and ask a question from each topic: services, pricing, climate claims, contact.
2. Check that every answer matches the source documents, especially numbers such as the carbon reduction figure.
3. Ask something the documents do not cover and confirm the assistant says it does not know and points to the contact page.
4. Call the API from the website's domain and confirm CORS allows it, and that another origin is blocked.
5. Open the Vercel preview deployment for the branch and repeat the checks.

## Adding or changing FAQ content

1. Add or replace the PDF or page in [the folder or list your code reads].
2. Keep one topic per document and use clear headings so the right text is found.
3. Run the checks above, then open a pull request.

## Accuracy and data protection

- The assistant gives general information about UmbrellaLabs. It is not a contract, quote or guarantee. Pricing and ROI commitments are agreed in writing with the sales team.
- Questions are sent to Google's Gemini API. Say so in your privacy policy, and tell visitors not to enter sensitive personal details.
- Do not log full questions with personal data longer than you need to. Check the data protection laws that apply to you.
- Keep API keys out of source control and out of browser code.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| Vercel build fails | A package is missing from `requirements.txt`, or the Python version is not supported. Check the build log |
| `/api/...` returns 404 | The route is not defined in `api/index.py`, or the path does not start with `/api/` |
| The home page returns 404 | `vercel.json` only routes `/api/(.*)`. Add a catch-all route to `api/index.py` and include `templates/` in the build with `includeFiles` if the app serves pages |
| Browser shows a CORS error | The calling site's origin is not allowed. Set the allowed origins and redeploy |
| Every answer is an error | The Gemini key is missing or wrong in Vercel, or the project was not redeployed after adding it |
| Answers are slow or time out | Source text is too long, or the function duration is too low. Shorten the text and raise the duration within your plan |
| Works locally but not on Vercel | The code reads a local file that was not deployed, or relies on a package not listed in `requirements.txt` |
| PDF text comes out empty | The PDF is a scan with no text layer. Use a text PDF or run OCR first |

## Contributing

Issues and pull requests are welcome. Please:

- describe what you changed and how you tested it
- keep FAQ wording plain, accurate and consistent with the published site
- never commit keys, `.env` files or personal data

For security issues, contact [security contact email] instead of opening a public issue.

## License

[Choose a license and add a `LICENSE` file. State it here.]

Copyright (C) 2026 UmbrellaNET
