def extract_page_properties(properties):
    """Extract non-title properties from a Notion page and render as markdown."""
    if not properties:
        return ""

    sections = []
    for prop_name, prop in properties.items():
        prop_type = prop.get("type", "")
        if prop_type == "title":
            continue

        value = _extract_property_value(prop, prop_type)
        if not value:
            continue

        sections.append(f"**{prop_name}:** {value}")

    return "\n\n".join(sections)


def extract_page_properties_structured(properties):
    """Extract non-title properties as a dict of {name: value} pairs (for frontmatter)."""
    if not properties:
        return {}

    result = {}
    for prop_name, prop in properties.items():
        prop_type = prop.get("type", "")
        if prop_type == "title":
            continue

        value = _extract_property_value(prop, prop_type)
        if value:
            result[prop_name] = value

    return result


def _extract_property_value(prop, prop_type):
    """Extract a human-readable value from a Notion property based on its type."""
    data = prop.get(prop_type)
    if data is None:
        return ""

    if prop_type == "rich_text":
        return _rich_text_to_markdown(data)

    if prop_type == "multi_select":
        names = [item.get("name", "") for item in data if item.get("name")]
        return ", ".join(names)

    if prop_type == "select":
        return data.get("name", "") if data else ""

    if prop_type == "status":
        return data.get("name", "") if data else ""

    if prop_type == "number":
        return str(data) if data is not None else ""

    if prop_type == "checkbox":
        return "Yes" if data else "No"

    if prop_type == "date":
        if not data:
            return ""
        start = data.get("start", "")
        end = data.get("end", "")
        return f"{start} - {end}" if end else start

    if prop_type == "url":
        return f"[{data}]({data})" if data else ""

    if prop_type in ("email", "phone_number"):
        return str(data) if data else ""

    if prop_type in ("created_by", "last_edited_by"):
        return data.get("name", "") if data else ""

    if prop_type in ("created_time", "last_edited_time"):
        return str(data) if data else ""

    if prop_type == "people":
        names = [p.get("name", "") for p in data if p.get("name")]
        return ", ".join(names)

    if prop_type == "files":
        links = []
        for f in data:
            name = f.get("name", "file")
            file_data = f.get(f.get("type", ""), {})
            url = file_data.get("url", "")
            if url:
                links.append(f"[{name}]({url})")
        return ", ".join(links)

    if prop_type == "relation":
        titles = [r.get("title", "") for r in data if r.get("title")]
        return ", ".join(titles)

    if prop_type == "rollup":
        return _extract_rollup_value(data)

    if prop_type == "formula":
        return _extract_formula_value(data)

    if prop_type == "unique_id":
        prefix = data.get("prefix", "")
        number = data.get("number", "")
        return f"{prefix}-{number}" if prefix else str(number)

    # Unknown property type — try string representation
    if isinstance(data, str):
        return data
    return ""


def _extract_rollup_value(data):
    """Extract value from a rollup property."""
    if not data:
        return ""
    rollup_type = data.get("type", "")
    if rollup_type == "number":
        val = data.get("number")
        return str(val) if val is not None else ""
    if rollup_type == "date":
        date_val = data.get("date")
        if date_val:
            start = date_val.get("start", "")
            end = date_val.get("end", "")
            return f"{start} - {end}" if end else start
        return ""
    if rollup_type == "array":
        items = data.get("array", [])
        values = []
        for item in items:
            item_type = item.get("type", "")
            val = _extract_property_value(item, item_type)
            if val:
                values.append(val)
        return ", ".join(values)
    return ""


def _extract_formula_value(data):
    """Extract value from a formula property."""
    if not data:
        return ""
    formula_type = data.get("type", "")
    val = data.get(formula_type)
    if val is None:
        return ""
    if formula_type == "boolean":
        return "Yes" if val else "No"
    if formula_type == "date":
        start = val.get("start", "") if isinstance(val, dict) else str(val)
        return start
    return str(val)


def convert_blocks_to_markdown(blocks, depth=0, max_depth=10):
    if depth >= max_depth:
        return ""

    lines = []
    numbered_counter = 0

    for block in blocks:
        block_type = block.get("type", "")

        if block_type == "numbered_list_item":
            numbered_counter += 1
        else:
            numbered_counter = 0

        line = _convert_block(block, depth, max_depth, numbered_counter)
        if line is not None:
            lines.append(line)

    return "\n".join(lines)


def _convert_block(block, depth, max_depth, numbered_counter):
    block_type = block.get("type", "")
    data = block.get(block_type, {})
    children = block.get("children", [])

    if block_type == "paragraph":
        text = _rich_text_to_markdown(data.get("rich_text", []))
        return _with_children(text, children, depth, max_depth)

    if block_type in ("heading_1", "heading_2", "heading_3"):
        level = int(block_type[-1])
        prefix = "#" * level
        text = _rich_text_to_markdown(data.get("rich_text", []))
        return f"{prefix} {text}"

    if block_type == "bulleted_list_item":
        text = _rich_text_to_markdown(data.get("rich_text", []))
        indent = "  " * depth
        result = f"{indent}- {text}"
        if children:
            child_md = convert_blocks_to_markdown(children, depth + 1, max_depth)
            if child_md:
                result += "\n" + child_md
        return result

    if block_type == "numbered_list_item":
        text = _rich_text_to_markdown(data.get("rich_text", []))
        indent = "  " * depth
        result = f"{indent}{numbered_counter}. {text}"
        if children:
            child_md = convert_blocks_to_markdown(children, depth + 1, max_depth)
            if child_md:
                result += "\n" + child_md
        return result

    if block_type == "to_do":
        text = _rich_text_to_markdown(data.get("rich_text", []))
        checked = data.get("checked", False)
        checkbox = "[x]" if checked else "[ ]"
        return f"- {checkbox} {text}"

    if block_type == "code":
        text = _rich_text_to_markdown(data.get("rich_text", []), apply_annotations=False)
        language = data.get("language", "")
        return f"```{language}\n{text}\n```"

    if block_type == "quote":
        text = _rich_text_to_markdown(data.get("rich_text", []))
        lines = text.split("\n")
        quoted = "\n".join(f"> {line}" for line in lines)
        return _with_children(quoted, children, depth, max_depth)

    if block_type == "callout":
        icon = data.get("icon") or {}
        emoji = icon.get("emoji", "") if icon.get("type") == "emoji" else ""
        text = _rich_text_to_markdown(data.get("rich_text", []))
        prefix = f"{emoji} " if emoji else ""
        result = f"> {prefix}{text}"
        if children:
            child_md = convert_blocks_to_markdown(children, depth, max_depth)
            if child_md:
                child_lines = child_md.split("\n")
                result += "\n" + "\n".join(f"> {line}" for line in child_lines)
        return result

    if block_type == "divider":
        return "---"

    if block_type == "image":
        image_data = data.get(data.get("type", ""), {})
        url = image_data.get("url", "")
        caption = _rich_text_to_markdown(data.get("caption", []))
        alt = caption if caption else "image"
        return f"![{alt}]({url})"

    if block_type == "bookmark":
        url = data.get("url", "")
        caption = _rich_text_to_markdown(data.get("caption", []))
        label = caption if caption else url
        return f"[{label}]({url})"

    if block_type == "embed":
        url = data.get("url", "")
        return f"[Embed: {url}]({url})"

    if block_type == "table":
        return _convert_table(children)

    if block_type == "toggle":
        text = _rich_text_to_markdown(data.get("rich_text", []))
        result = f"**{text}**"
        if children:
            child_md = convert_blocks_to_markdown(children, depth, max_depth)
            if child_md:
                result += "\n" + child_md
        return result

    if block_type == "child_page":
        title = data.get("title", "")
        return f"[Child page: {title}]"

    if block_type == "child_database":
        title = data.get("title", "")
        return f"[Child database: {title}]"

    if block_type == "equation":
        expression = data.get("expression", "")
        return f"$${expression}$$"

    if block_type in ("synced_block", "column_list", "column"):
        if children:
            return convert_blocks_to_markdown(children, depth, max_depth)
        return None

    if block_type == "table_of_contents":
        return None

    if block_type == "breadcrumb":
        return None

    if block_type == "link_preview":
        url = data.get("url", "")
        return f"[Link: {url}]({url})" if url else None

    if block_type == "pdf":
        pdf_data = data.get(data.get("type", ""), {})
        url = pdf_data.get("url", "")
        caption = _rich_text_to_markdown(data.get("caption", []))
        return f"[PDF: {caption or url}]({url})"

    if block_type == "video":
        video_data = data.get(data.get("type", ""), {})
        url = video_data.get("url", "")
        caption = _rich_text_to_markdown(data.get("caption", []))
        return f"[Video: {caption or url}]({url})"

    if block_type == "file":
        file_data = data.get(data.get("type", ""), {})
        url = file_data.get("url", "")
        caption = _rich_text_to_markdown(data.get("caption", []))
        return f"[File: {caption or url}]({url})"

    # Unknown block type: try to extract rich_text content
    rich_text = data.get("rich_text", [])
    if rich_text:
        return _rich_text_to_markdown(rich_text)

    return f"[Unsupported: {block_type}]"


def _with_children(text, children, depth, max_depth):
    if not children:
        return text
    child_md = convert_blocks_to_markdown(children, depth, max_depth)
    if child_md:
        return text + "\n" + child_md if text else child_md
    return text


def _convert_table(rows):
    if not rows:
        return ""

    table_rows = []
    for row in rows:
        row_data = row.get("table_row", {})
        cells = row_data.get("cells", [])
        cell_texts = [_rich_text_to_markdown(cell) for cell in cells]
        table_rows.append("| " + " | ".join(cell_texts) + " |")

    if len(table_rows) >= 1:
        num_cols = len(rows[0].get("table_row", {}).get("cells", []))
        separator = "| " + " | ".join(["---"] * num_cols) + " |"
        table_rows.insert(1, separator)

    return "\n".join(table_rows)


def _rich_text_to_markdown(rich_text_array, apply_annotations=True):
    parts = []
    for rt in rich_text_array:
        text = rt.get("plain_text", "")
        if not text:
            continue

        if apply_annotations:
            annotations = rt.get("annotations", {})
            if annotations.get("code"):
                text = f"`{text}`"
            if annotations.get("bold"):
                text = f"**{text}**"
            if annotations.get("italic"):
                text = f"*{text}*"
            if annotations.get("strikethrough"):
                text = f"~~{text}~~"

        href = rt.get("href")
        if href and apply_annotations:
            text = f"[{text}]({href})"

        parts.append(text)

    return "".join(parts)
