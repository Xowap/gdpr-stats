# GDPR Statistics Tool

This tool will extract data from the CNIL website to know exactly what is being
done with GDPR.

To run you need to have the following environment variables set:

-   `ANTHROPIC_API_KEY` &mdash; API key for the Anthropic API

This can also be done via a `.env` file at the root of the project.

To run the project simply do:

```
.venv/bin/python -m gdpr_stats output.yaml
```

This will create a `output.yaml` file which you can then import from any Python
script in order to crunch the data.
