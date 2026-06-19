"""Generate ComfyUI UI-format workflows (drag-and-drop loadable) for DiT360.

Run: python workflows/_build_workflows.py
Produces dit360_t2p.json and dit360_edit.json next to this file.

Kept in-repo so the graphs can be regenerated if node signatures change.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self._nid = 0
        self._lid = 0

    def add(self, type, pos, widgets=None, inputs=None, outputs=None, size=(260, 120)):
        self._nid += 1
        node = {
            "id": self._nid, "type": type, "pos": list(pos), "size": list(size),
            "flags": {}, "order": self._nid - 1, "mode": 0,
            "inputs": [dict(name=n, type=t, link=None) for n, t in (inputs or [])],
            "outputs": [dict(name=n, type=t, links=[], slot_index=i)
                        for i, (n, t) in enumerate(outputs or [])],
            "properties": {"Node name for S&R": type},
            "widgets_values": widgets or [],
        }
        self.nodes.append(node)
        return node

    def link(self, src, src_slot, dst, dst_slot):
        self._lid += 1
        ltype = src["outputs"][src_slot]["type"]
        self.links.append([self._lid, src["id"], src_slot, dst["id"], dst_slot, ltype])
        src["outputs"][src_slot]["links"].append(self._lid)
        dst["inputs"][dst_slot]["link"] = self._lid
        return self._lid

    def dump(self, path):
        wf = {
            "last_node_id": self._nid, "last_link_id": self._lid,
            "nodes": self.nodes, "links": self.links,
            "groups": [], "config": {}, "extra": {}, "version": 0.4,
        }
        with open(path, "w") as f:
            json.dump(wf, f, indent=2)
        # integrity check: every input.link and output.links id exists
        ids = {l[0] for l in self.links}
        for n in self.nodes:
            for inp in n["inputs"]:
                assert inp["link"] is None or inp["link"] in ids, (n["type"], inp)
            for out in n["outputs"]:
                for l in out["links"]:
                    assert l in ids
        print("wrote", path, f"({self._nid} nodes, {self._lid} links)")


LOADER_W = [
    "⤓ download fp8", "⤓ download fp8", "fp8_e4m3fn", 1.0,
    "Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors",
    "Insta360-Research/DiT360-Panorama-Image-Generation", "adapter_model.safetensors",
]
LOADER_OUT = [("model", "MODEL"), ("clip", "CLIP"), ("vae", "VAE")]


def build_t2p():
    g = Graph()
    loader = g.add("DiT360ModelLoader", (40, 200), LOADER_W,
                   outputs=LOADER_OUT, size=(340, 220))
    pos = g.add("CLIPTextEncode", (430, 110),
                ["This is a panorama image. A medieval castle stands proudly on a hilltop "
                 "surrounded by autumn forests, with golden light spilling across the landscape."],
                [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], size=(380, 160))
    neg = g.add("CLIPTextEncode", (430, 330), [""],
                [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], size=(380, 110))
    guid = g.add("FluxGuidance", (850, 110), [2.8],
                 [("conditioning", "CONDITIONING")], [("CONDITIONING", "CONDITIONING")],
                 size=(240, 60))
    lat = g.add("EmptySD3LatentImage", (430, 500), [2048, 1024, 1],
                outputs=[("LATENT", "LATENT")], size=(280, 110))
    smp = g.add("DiT360PanoramaSampler", (1130, 200),
                [0, "fixed", 28, 1.0, "euler", "simple", 1.0, True, 1],
                [("model", "MODEL"), ("positive", "CONDITIONING"),
                 ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                [("LATENT", "LATENT")], size=(300, 320))
    dec = g.add("VAEDecode", (1480, 200), [],
                [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], size=(220, 60))
    save = g.add("SaveImage", (1740, 200), ["DiT360_panorama"],
                 [("images", "IMAGE")], size=(360, 320))

    g.link(loader, 1, pos, 0)
    g.link(loader, 1, neg, 0)
    g.link(pos, 0, guid, 0)
    g.link(loader, 0, smp, 0)
    g.link(guid, 0, smp, 1)
    g.link(neg, 0, smp, 2)
    g.link(lat, 0, smp, 3)
    g.link(smp, 0, dec, 0)
    g.link(loader, 2, dec, 1)
    g.link(dec, 0, save, 0)
    g.dump(os.path.join(HERE, "dit360_t2p.json"))


def build_edit():
    g = Graph()
    loader = g.add("DiT360ModelLoader", (40, 200), LOADER_W,
                   outputs=LOADER_OUT, size=(340, 220))
    img = g.add("LoadImage", (40, 500), ["chalet_panorama.png", "image"],
                outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], size=(300, 360))
    enc = g.add("VAEEncode", (430, 540), [],
                [("pixels", "IMAGE"), ("vae", "VAE")], [("LATENT", "LATENT")], size=(240, 60))
    src = g.add("CLIPTextEncode", (430, 110), ["This is a panorama image."],
                [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], size=(380, 110))
    edit = g.add("CLIPTextEncode", (430, 280),
                 ["This is a panorama image. The image depicts a village next to a snow-capped mountain"],
                 [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")], size=(380, 130))
    inv = g.add("DiT360RFInvert", (870, 380), [28, "simple", 0.5, 0, "fixed"],
                [("model", "MODEL"), ("source_conditioning", "CONDITIONING"),
                 ("latent_image", "LATENT")],
                [("inversion", "DIT360_INVERSION")], size=(300, 180))
    op = g.add("DiT360Outpaint", (1230, 200),
               [2048, 1024, 0.5, 1.0, 0.0, 0.99],
               [("model", "MODEL"), ("inversion", "DIT360_INVERSION"),
                ("edit_conditioning", "CONDITIONING"), ("mask", "MASK")],
               [("LATENT", "LATENT")], size=(300, 280))
    dec = g.add("VAEDecode", (1580, 200), [],
                [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], size=(220, 60))
    save = g.add("SaveImage", (1840, 200), ["DiT360_edit"],
                 [("images", "IMAGE")], size=(360, 320))

    g.link(loader, 1, src, 0)
    g.link(loader, 1, edit, 0)
    g.link(img, 0, enc, 0)
    g.link(loader, 2, enc, 1)
    g.link(loader, 0, inv, 0)
    g.link(src, 0, inv, 1)
    g.link(enc, 0, inv, 2)
    g.link(loader, 0, op, 0)
    g.link(inv, 0, op, 1)
    g.link(edit, 0, op, 2)
    g.link(img, 1, op, 3)
    g.link(op, 0, dec, 0)
    g.link(loader, 2, dec, 1)
    g.link(dec, 0, save, 0)
    g.dump(os.path.join(HERE, "dit360_edit.json"))


if __name__ == "__main__":
    build_t2p()
    build_edit()
