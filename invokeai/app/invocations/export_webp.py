import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext, InputField, invocation
from invokeai.app.invocations.metadata import ImageMetadata
from invokeai.app.invocations.primitives import ImageField, ImageOutput

XMP_NS_URI = "https://invoke.ai/xmp/0.0.1/"
XMP_NS_PREFIX = "invokeai"


class MetadataXMPSerializer:
    """Export ImageMetadata as a Extensible Metadata Platform (XMP) description.

    XMP can be embedded in a wide variety of media formats.

    https://developer.adobe.com/xmp/docs/
    """

    _XML_NAMESPACES: dict[str, str] = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
        "xmp": "http://ns.adobe.com/xap/1.0/",
        XMP_NS_PREFIX: XMP_NS_URI,
    }

    def get_xmp_data(self, metadata: ImageMetadata) -> dict:
        create_date = datetime.now(timezone.utc)
        xmp_data = {
            # XMP Basic
            "xmp:CreatorTool": self.xmp_creator_tool(),
            "xmp:CreateDate": f"{create_date:%Y-%m-%dT%H:%M:%S.%f%z}",
            # "xmp:Label": "",  # TODO: text
            # "xmp:Identifier": [""],  # TODO: array of text
            # "xmp:Thumbnails": "",  # TODO
        }

        if metadata.metadata is not None:
            xmp_data[f"{XMP_NS_PREFIX}:metadata"] = json.dumps(metadata.metadata)
        if metadata.graph is not None:
            # TODO: graph minimizer, because 36-character node IDs are way too fat.
            xmp_data[f"{XMP_NS_PREFIX}:graph"] = json.dumps(metadata.graph)

        return xmp_data

    def get_xmp(self, metadata: ImageMetadata) -> str:
        return self.encode_xmp(self.get_xmp_data(metadata))

    def xmp_creator_tool(self) -> str:
        # XMP Specification Part 1, Section 8.2.2.1:
        #   Organization Software Version (token;token;…)
        # Organization can't have spaces but Software can? weird.
        from invokeai import version

        return f"invoke.ai {version.__app_name__ } {version.__version__}"

    def encode_xmp(self, xmp_data: dict) -> str:
        # This should be using an RDF serializer, but maybe we have few enough fields that we can fake it?
        for prefix, uri in self._XML_NAMESPACES.items():
            ET.register_namespace(prefix, uri)

        root = ET.Element("rdf:RDF")
        desc = ET.SubElement(root, "rdf:Description", about="")

        # We have to expand the namespace prefix for the Element constructor.
        for key, value in xmp_data.items():
            ns, tag = key.split(":", 1)
            full_key = f"{{{self._XML_NAMESPACES[ns]}}}{tag}"
            ET.SubElement(desc, full_key).text = value

        return ET.tostring(root, encoding="unicode")

    def get_exif(self, metadata: ImageMetadata) -> bytes:
        return self.encode_exif(self.get_exif_data(metadata))

    def get_exif_data(self, metadata: ImageMetadata) -> dict:
        raise NotImplementedError()  # return {}

    def encode_exif(self, exif_data: dict) -> bytes:
        # needs an exif encoder, and piexif needs maintenance help: https://github.com/JEFuller/Piexif
        raise NotImplementedError()  # return bytes()


@invocation("export_image", title="Export Image (WebP)", tags=["image"], category="image")
class WebExportInvocation(BaseInvocation):
    """Export an image in WebP format."""

    # Inputs
    image: ImageField = InputField(default=None, description="The image to export")
    lossless: bool = Field(default=False, description="Use lossless compression.")
    quality: int = Field(
        default=80, description="Image quality [0..100]. 0=smallest file, 100=highest quality", ge=0, le=100
    )

    # internals
    _metadataExporter = MetadataXMPSerializer()
    _format = "WEBP"

    def invoke(self, context: InvocationContext) -> ImageOutput:
        images = context.services.images
        name = self.image.image_name
        image = images.get_pil_image(name)

        # TODO: this should use a service to write, instead of direct filesystem access

        metadata: ImageMetadata = images.get_metadata(name)

        image_path = self.get_path(name)
        image.save(
            image_path,
            self._format,
            lossless=self.lossless,
            quality=self.quality,
            # exif=self._metadataExporter.get_exif(metadata),
            xmp=self._metadataExporter.get_xmp(metadata),
        )

        context.services.logger.info(f"saved to {image_path}")

        return ImageOutput(
            image=ImageField(image_name=self.image.image_name),
            width=image.width,
            height=image.height,
        )

    def get_path(self, image_name: str) -> Path:
        # FIXME: should be in the right directory
        return Path(image_name).with_suffix(".webp")
