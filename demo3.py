import argparse
import base64
import json
import urllib.request


def load_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Hunyuan3D API /generate")
    parser.add_argument("--host", default="127.0.0.1:10083", help="host:port for API")
    parser.add_argument("--image", default="assets/demo.png", help="input image path")
    parser.add_argument("--out", default="out.glb", help="output glb path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    payload = {
        "image": load_base64_image(args.image),
        "texture": False,
        "seed": args.seed,
    }

    req = urllib.request.Request(
        f"http://{args.host}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    with open(args.out, "wb") as f:
        f.write(data)

    print(f"saved: {args.out}, size: {len(data)}")


if __name__ == "__main__":
    main()
