from mcp.server.mcpserver import MCPServer

from badil import service

INSTRUCTIONS = (
    "Badil helps people in Tunisia find safe alternatives to medicines that are out of stock or too expensive. "
    "Each tool does its whole job in one call, so call one tool per user need. "
    "Always show the citations and the 'requires_validation' field (pharmacist or doctor) to the user, "
    "never present a yellow or orange substitute as a direct swap, and never invent medical facts beyond the tool output. "
    "If a tool returns status 'did_you_mean', ask the user to choose among the suggestions."
)

mcp = MCPServer(name="badil", instructions=INSTRUCTIONS)


@mcp.tool()
def find_substitute(medicine: str, pregnant: bool = False, breastfeeding: bool = False,
                    age_years: int | None = None, allergies: list[str] | None = None) -> dict:
    """Find the safest available substitute for a medicine (out of stock, or a cheaper strict equivalent when in stock).

    Accepts a brand name, an active ingredient, a typo or Arabic script. Optional patient context (pregnancy,
    breastfeeding, age, allergies) is checked against the substitute's official leaflet; contraindicated
    substitutes are excluded. Returns a verdict with risk tier (green = the pharmacist can dispense it unless the
    prescription says "non substituable", yellow = pharmacist validates, orange = doctor validates, red = refused),
    who must validate, safety notes, arguments for and against, alternatives and dated citations.
    """
    profile = {"pregnant": pregnant, "breastfeeding": breastfeeding, "age_years": age_years,
               "allergies": allergies or []}
    has_context = pregnant or breastfeeding or age_years is not None or bool(allergies)
    return service.find_substitute(medicine, profile if has_context else None)


@mcp.tool()
def lookup_medicine(medicine: str) -> dict:
    """Show a medicine's card: active ingredient, dose, form, price, live availability, CNAM prior authorization,
    indication and its other presentations (doses and box sizes)."""
    return service.lookup_medicine(medicine)


@mcp.tool()
def cheaper_alternatives(medicine: str, max_price_dt: float | None = None) -> dict:
    """List in-stock strict equivalents (same active ingredient, dose and form) that are cheaper per unit,
    optionally under a maximum box price in Tunisian dinars."""
    return service.cheaper_alternatives(medicine, max_price_dt)


@mcp.tool()
def medicines_for_symptom(question: str) -> dict:
    """Find medicines whose official indication matches a symptom or condition, described in French or Arabic.
    In-stock medicines come first. Informational only: prescription medicines still require a doctor."""
    return service.medicines_for_symptom(question)


@mcp.tool()
def leaflet_section(medicine: str, section: service.Section) -> dict:
    """Return one section of a medicine's official leaflet: posology, contraindications, warnings,
    interactions, pregnancy (fertility, pregnancy and breastfeeding), side_effects or overdose."""
    return service.leaflet_section(medicine, section)


@mcp.tool()
def ask_leaflet(medicine: str, question: str) -> dict:
    """Answer a specific question about one medicine (pregnancy, alcohol, children, missed dose...)
    with the most relevant passages of its official leaflet."""
    return service.ask_leaflet(medicine, question)


if __name__ == "__main__":
    service.warm_up()
    mcp.run("stdio")
