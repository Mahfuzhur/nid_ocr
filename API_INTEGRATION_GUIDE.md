# NID OCR API — Integration Guide

Version: 3.0

This API extracts printed data fields from images of Bangladesh National ID
(NID) cards. Send an image of the front or back of a card and receive the
extracted fields back as JSON.

## Demo Environment

```
Base URL: http://35.238.26.55:8000
```

This environment is provided for integration testing. It is not guaranteed to
retain uptime SLAs; a production endpoint with authentication and SLAs can be
provisioned separately once integration is confirmed.

- Interactive API explorer: `http://35.238.26.55:8000/docs`

## Authentication

No authentication is required in this demo environment. Production access
will be issued with an API key or token — this section will be updated once
that is provisioned. Until then, please treat the demo URL as confidential
and do not share it outside your integration team.

---

## Endpoints

### `POST /nid/front`

Extracts fields from the **front side** of an NID card.

**Request** — `multipart/form-data`

| Field  | Type | Required | Description                        |
| ------ | ---- | -------- | ---------------------------------- |
| `file` | file | yes      | Image of the front of the NID card |

Supported file types: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

**Response** — `200 OK`, `application/json`

```json
{
  "name": "MD SAKIB HASAN",
  "father_name": "MD ABDUL KARIM",
  "mother_name": "MST RAHIMA BEGUM",
  "spouse_name": null,
  "date_of_birth": "01 Jan 1995",
  "nid_number": "1234567890123"
}
```

| Field           | Type           | Notes                             |
| --------------- | -------------- | --------------------------------- |
| `name`          | string \| null | Cardholder's name                 |
| `father_name`   | string \| null |                                   |
| `mother_name`   | string \| null |                                   |
| `spouse_name`   | string \| null | Present only on some card layouts |
| `date_of_birth` | string \| null | As printed on the card            |
| `nid_number`    | string \| null |                                   |

Fields not found on the card are returned as `null` — this is expected
behavior for cards that omit a field (e.g. no spouse name), not necessarily
an error.

**Example**

```bash
curl -X POST "http://35.238.26.55:8000/nid/front" \
  -F "file=@front.jpg"
```

---

### `POST /nid/back`

Extracts fields from the **back side** of an NID card.

**Request** — `multipart/form-data`

| Field  | Type | Required | Description                       |
| ------ | ---- | -------- | --------------------------------- |
| `file` | file | yes      | Image of the back of the NID card |

Supported file types: same as `/nid/front`.

**Response** — `200 OK`, `application/json`

```json
{
  "address": "House/Holding: 535 ..., Village/Road: ..., Post Office: ..., Dhaka",
  "blood_group": "B+",
  "issue_date": "26/08/2021",
  "place_of_birth": "Dhaka"
}
```

| Field            | Type           | Notes                                        |
| ---------------- | -------------- | -------------------------------------------- |
| `address`        | string \| null |                                              |
| `blood_group`    | string \| null |                                              |
| `issue_date`     | string \| null |                                              |
| `place_of_birth` | string \| null | Present only on newer ("Smart") card layouts |

**Example**

```bash
curl -X POST "http://35.238.26.55:8000/nid/back" \
  -F "file=@back.jpg"
```

---

## Error Responses

Errors are returned as JSON: `{"detail": "<message>"}`.

| Status                      | Meaning                     | Typical cause                                                                                                                                                            |
| --------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `400 Bad Request`           | Invalid request             | The uploaded file's type isn't one of the supported image formats.                                                                                                       |
| `422 Unprocessable Entity`  | Could not process the image | The image was readable but the system could not recognize it as an NID card, or could not extract usable data from it. Usually resolved by re-uploading a clearer photo. |
| `500 Internal Server Error` | Unexpected server error     | Transient failure — safe to retry. Contact us if it persists.                                                                                                            |

---

## Integration Notes

- **One request per side.** The front and back of a card are two separate uploads to two separate endpoints — there is no combined endpoint.
- **Image quality matters.** Well-lit, in-focus, uncropped photos taken directly above the card produce the most reliable results. Avoid glare and shadows.
- **Missing fields are not failures.** A `null` value means that particular field wasn't present or legible on the card, not that the request failed — check the HTTP status code to determine success/failure, not whether every field is populated.
- **Idempotent and stateless.** Each request is processed independently; there is no session or multi-step flow.

---

## Support

For integration support please contact us directly.
