//! Randomized differential tests: the GADDAG generator against the CSP port
//! of solver.py, on random games over a small fixed dictionary (all 2- and
//! 3-letter SOWPODS words plus 700 longer ones), played under the full
//! rules and then replayed with real racks. `bingo verify` runs the same
//! comparison with the full word list.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const csp = @import("csp.zig");
const game = @import("game.zig");

const Move = board.Move;
const Position = board.Position;

const test_words =
    \\AA AB AD AE AG AH AI AL AM AN AR AS AT AW
    \\AX AY BA BE BI BO BY CH DA DE DI DO EA ED
    \\EE EF EH EL EM EN ER ES ET EX FA FE FY GI
    \\GO GU HA HE HI HM HO ID IF IN IO IS IT JA
    \\JO KA KI KO KY LA LI LO MA ME MI MM MO MU
    \\MY NA NE NO NU NY OB OD OE OF OH OI OM ON
    \\OO OP OR OS OU OW OX OY PA PE PI PO QI RE
    \\SH SI SO ST TA TE TI TO UG UH UM UN UP UR
    \\US UT WE WO XI XU YA YE YO YU ZA ZO AAH AAL
    \\AAS ABA ABB ABO ABS ABY ACE ACH ACT ADD ADO ADS ADZ AFF
    \\AFT AGA AGE AGO AGS AHA AHI AHS AIA AID AIL AIM AIN AIR
    \\AIS AIT AKA AKE ALA ALB ALE ALF ALL ALP ALS ALT AMA AMI
    \\AMP AMU ANA AND ANE ANI ANN ANT ANY APE APO APP APT ARB
    \\ARC ARD ARE ARF ARK ARM ARS ART ARY ASH ASK ASP ASS ATE
    \\ATT AUA AUE AUF AUK AVA AVE AVO AWA AWE AWL AWN AXE AYE
    \\AYS AYU AZO BAA BAC BAD BAG BAH BAL BAM BAN BAP BAR BAS
    \\BAT BAY BED BEE BEG BEL BEN BES BET BEY BEZ BIB BID BIG
    \\BIN BIO BIS BIT BIZ BOA BOB BOD BOG BOH BOI BOK BON BOO
    \\BOP BOR BOS BOT BOW BOX BOY BRA BRO BRR BRU BUB BUD BUG
    \\BUM BUN BUR BUS BUT BUY BYE BYS CAA CAB CAD CAG CAM CAN
    \\CAP CAR CAT CAW CAY CAZ CEE CEL CEP CHA CHE CHI CID CIG
    \\CIS CIT CLY COB COD COG COL CON COO COP COR COS COT COW
    \\COX COY COZ CRU CRY CUB CUD CUE CUM CUP CUR CUT CUZ CWM
    \\DAB DAD DAE DAG DAH DAK DAL DAM DAN DAP DAS DAW DAY DEB
    \\DEE DEF DEG DEI DEL DEN DEV DEW DEX DEY DIB DID DIE DIF
    \\DIG DIM DIN DIP DIS DIT DIV DOB DOC DOD DOE DOF DOG DOH
    \\DOL DOM DON DOO DOP DOR DOS DOT DOW DOY DRY DSO DUB DUD
    \\DUE DUG DUH DUI DUN DUO DUP DUX DYE DZO EAN EAR EAS EAT
    \\EAU EBB ECH ECO ECU EDH EDS EEK EEL EEN EFF EFS EFT EGG
    \\EGO EHS EIK EKE ELD ELF ELK ELL ELM ELS ELT EME EMO EMS
    \\EMU END ENE ENG ENS EON ERA ERE ERF ERG ERK ERN ERR ERS
    \\ESS EST ETA ETH EUK EVE EVO EWE EWK EWT EXO EYE FAA FAB
    \\FAD FAE FAG FAH FAN FAP FAR FAS FAT FAW FAX FAY FED FEE
    \\FEG FEH FEM FEN FER FES FET FEU FEW FEY FEZ FIB FID FIE
    \\FIG FIL FIN FIR FIT FIX FIZ FLU FLY FOB FOE FOG FOH FON
    \\FOP FOR FOU FOX FOY FRA FRO FRY FUB FUD FUG FUM FUN FUR
    \\GAB GAD GAE GAG GAL GAM GAN GAP GAR GAS GAT GAU GAY GED
    \\GEE GEL GEM GEN GEO GET GEY GHI GIB GID GIE GIF GIG GIN
    \\GIO GIP GIS GIT GJU GNU GOA GOB GOD GOE GON GOO GOR GOS
    \\GOT GOV GOX GOY GUB GUE GUL GUM GUN GUP GUR GUS GUT GUV
    \\GUY GYM GYP HAD HAE HAG HAH HAJ HAM HAN HAO HAP HAS HAT
    \\HAW HAY HEH HEM HEN HEP HER HES HET HEW HEX HEY HIC HID
    \\HIE HIM HIN HIP HIS HIT HMM HOA HOB HOC HOD HOE HOG HOH
    \\HOI HOM HON HOO HOP HOS HOT HOW HOX HOY HUB HUE HUG HUH
    \\HUI HUM HUN HUP HUT HYE HYP ICE ICH ICK ICY IDE IDS IFF
    \\IFS IGG ILK ILL IMP INK INN INS ION IOS IRE IRK ISH ISM
    \\ISO ITA ITS IVY IWI JAB JAG JAI JAK JAM JAP JAR JAW JAY
    \\JEE JET JEU JEW JIB JIG JIN JIZ JOB JOE JOG JOL JOR JOT
    \\JOW JOY JUD JUG JUN JUS JUT KAB KAE KAF KAI KAK KAM KAS
    \\KAT KAW KAY KEA KEB KED KEF KEG KEN KEP KET KEX KEY KHI
    \\KID KIF KIN KIP KIR KIS KIT KOA KOB KOI KON KOP KOR KOS
    \\KOW KUE KYE KYU LAB LAC LAD LAG LAH LAM LAP LAR LAS LAT
    \\LAV LAW LAX LAY LEA LED LEE LEG LEI LEK LEP LES LET LEU
    \\LEV LEW LEX LEY LEZ LIB LID LIE LIG LIN LIP LIS LIT LOB
    \\LOD LOG LOO LOP LOR LOS LOT LOU LOW LOX LOY LUD LUG LUM
    \\LUR LUV LUX LUZ LYE LYM MAA MAC MAD MAE MAG MAK MAL MAM
    \\MAN MAP MAR MAS MAT MAW MAX MAY MED MEE MEG MEL MEM MEN
    \\MES MET MEU MEW MHO MIB MIC MID MIG MIL MIM MIR MIS MIX
    \\MIZ MNA MOA MOB MOC MOD MOE MOG MOI MOL MOM MON MOO MOP
    \\MOR MOS MOT MOU MOW MOY MOZ MUD MUG MUM MUN MUS MUT MUX
    \\MYC NAB NAE NAG NAH NAM NAN NAP NAS NAT NAW NAY NEB NED
    \\NEE NEF NEG NEK NEP NET NEW NIB NID NIE NIL NIM NIP NIS
    \\NIT NIX NOB NOD NOG NOH NOM NON NOO NOR NOS NOT NOW NOX
    \\NOY NTH NUB NUN NUR NUS NUT NYE NYS OAF OAK OAR OAT OBA
    \\OBE OBI OBO OBS OCA OCH ODA ODD ODE ODS OES OFF OFT OHM
    \\OHO OHS OIK OIL OKA OKE OLD OLE OLM OMS ONE ONO ONS ONY
    \\OOF OOH OOM OON OOP OOR OOS OOT OPE OPS OPT ORA ORB ORC
    \\ORD ORE ORF ORS ORT OSE OUD OUK OUP OUR OUS OUT OVA OWE
    \\OWL OWN OWT OXO OXY OYE OYS PAC PAD PAH PAL PAM PAN PAP
    \\PAR PAS PAT PAV PAW PAX PAY PEA PEC PED PEE PEG PEH PEN
    \\PEP PER PES PET PEW PHI PHO PHT PIA PIC PIE PIG PIN PIP
    \\PIR PIS PIT PIU PIX PLU PLY POA POD POH POI POL POM POO
    \\POP POS POT POW POX POZ PRE PRO PRY PSI PST PUB PUD PUG
    \\PUH PUL PUN PUP PUR PUS PUT PUY PYA PYE PYX QAT QIS QUA
    \\RAD RAG RAH RAI RAJ RAM RAN RAP RAS RAT RAW RAX RAY REB
    \\REC RED REE REF REG REH REI REM REN REO REP RES RET REV
    \\REW REX REZ RHO RHY RIA RIB RID RIF RIG RIM RIN RIP RIT
    \\RIZ ROB ROC ROD ROE ROK ROM ROO ROT ROW RUB RUC RUD RUE
    \\RUG RUM RUN RUT RYA RYE SAB SAC SAD SAE SAG SAI SAL SAM
    \\SAN SAP SAR SAT SAU SAV SAW SAX SAY SAZ SEA SEC SED SEE
    \\SEG SEI SEL SEN SER SET SEW SEX SEY SEZ SHA SHE SHH SHY
    \\SIB SIC SIF SIK SIM SIN SIP SIR SIS SIT SIX SKA SKI SKY
    \\SLY SMA SNY SOB SOC SOD SOG SOH SOL SOM SON SOP SOS SOT
    \\SOU SOV SOW SOX SOY SPA SPY SRI STY SUB SUD SUE SUI SUK
    \\SUM SUN SUP SUQ SUR SUS SWY SYE SYN TAB TAD TAE TAG TAI
    \\TAJ TAK TAM TAN TAO TAP TAR TAS TAT TAU TAV TAW TAX TAY
    \\TEA TEC TED TEE TEF TEG TEL TEN TES TET TEW TEX THE THO
    \\THY TIC TID TIE TIG TIL TIN TIP TIS TIT TIX TOC TOD TOE
    \\TOG TOM TON TOO TOP TOR TOT TOW TOY TRY TSK TUB TUG TUI
    \\TUM TUN TUP TUT TUX TWA TWO TWP TYE TYG UDO UDS UEY UFO
    \\UGH UGS UKE ULE ULU UMM UMP UMU UNI UNS UPO UPS URB URD
    \\URE URN URP USE UTA UTE UTS UTU UVA VAC VAE VAG VAN VAR
    \\VAS VAT VAU VAV VAW VEE VEG VET VEX VIA VID VIE VIG VIM
    \\VIN VIS VLY VOE VOL VOR VOW VOX VUG VUM WAB WAD WAE WAG
    \\WAI WAN WAP WAR WAS WAT WAW WAX WAY WEB WED WEE WEM WEN
    \\WET WEX WEY WHA WHO WHY WIG WIN WIS WIT WIZ WOE WOF WOG
    \\WOK WON WOO WOP WOS WOT WOW WOX WRY WUD WUS WYE WYN XIS
    \\YAD YAE YAG YAH YAK YAM YAP YAR YAW YAY YEA YEH YEN YEP
    \\YES YET YEW YEX YGO YID YIN YIP YOB YOD YOK YOM YON YOS
    \\YOU YOW YUG YUK YUM YUP YUS ZAG ZAP ZAS ZAX ZEA ZED ZEE
    \\ZEK ZEL ZEP ZEX ZHO ZIG ZIN ZIP ZIT ZIZ ZOA ZOL ZOO ZOS
    \\ZUZ ZZZ ABSORBEFACIENTS ACCUSTREMENT ACIDOPHILES ADIABATIC ADSORBS ADUSTING AEROGRAMMES AGUISES AIRSHAFT AIRWISE AJUGAS ALLANTOISES
    \\AMATIVENESSES AMBASSADRESS AMIDASES AMORTISATION AMPLIDYNE AMYGDALE ANAGOGICALLY ANDROMEDA ANGLICIZED ANISES ANNOYERS ANTHOCARP ANTICATHODES APERTNESS
    \\APIARIAN ARILLODE ARTEFACTS ASEXUALLY ASTIGMATICALLY ASTRONOMIZED AUDITORSHIPS AUSCULTATION AUTOPHOBIAS AUTOSTRADE BABELISMS BABLAH BADNESSES BAGELS
    \\BANGSRING BANOFFI BARBER BASENJIS BASSOS BATTELERS BAWCOCK BAYE BEDDER BEECHEN BELLYACHING BETOILS BETROTHED BETTERMENTS
    \\BICOMPONENT BIOSTROME BIRK BLUEBOOKS BOFF BRADYPEPTIC BROGUERY BRUSQUER BUDGEREE BULLSHITTED BUMBOAT BUMPTIOUSNESSES BYZANTINE CACTOBLASTIS
    \\CADDY CAMBOGE CANCAN CANCER CARAVANSERAIS CATLING CELLBLOCKS CEPE CEVAPCICI CHAMOIS CHAMPER CHARNECOS CHEESEBOARDS CHOANOCYTE
    \\CHOLERS CHORALIST CHROMATOGRAPH CICATRICIAL CINCTURED CLACKS CLANGOR CLASSIFYING CLAUSTROPHOBIA CLAUSULA CLAVICORN CLEANINGS CLISTOGAMY CLOBBER
    \\COACH COATDRESS COCAINIZE COCKHORSE COLLECTIBLES COLOPHONS COMFORTED COMFORTS COMPANDED COMPARTMENTS COMPASSIONATED COMPRIZING CONCESSIONIST CONFIGURATION
    \\CONJURER CONTUSIONED CONVIVIALITIES COPPERING CORDOBA COREGENTS COSTLINESS COUNTERBORING COWK COXCOMBICAL COYING CREDAL CRIMINALIZE CROCKERIES
    \\CROSSBENCHES CROSSNESSES CRUISIES CRYPTOLOGICAL CURVED CURVEDNESS CUSP CUSTOM CYMLINS CYPRIPEDIA CZARITZAS DATAMATIONS DECITIZENISED DEFILEMENTS
    \\DEFINIENTIA DEMANDINGNESSES DERIVATIVE DHAKS DHAL DIALECTICISMS DIGONEUTISM DIKDIK DILETTANTE DISCREPANCIES DISCRETEST DISPREAD DISTASTEFULNESS DISTAVES
    \\DISTRESSINGLY DITTIES DIVULSION DOES DOWNSHIFTINGS DREADNOUGHT DREARES DRYOPITHECINE DUUMVIRS DYSLOGISTICALLY ECHOGRAPHY EELS EFFEMINIZING EFFERE
    \\EFFERENT EFFETE ELECTROTYPERS EMOTERS EMPLUMES EMYS ENACTORS ENCALMING ENDOWER ENFORESTS ENLARGE ENMESHMENTS ENODAL ENTRAILS
    \\ENVYINGLY EQUALISATION EQUIPOLLENTLY ETERNALISED ETERNISING ETHICIZED EXCITATIONS EXCULPABLE EXHEDRA EXPLORING EXTEND FAKEER FASCINATINGLY FATHOMLESSNESS
    \\FECIAL FELDSPARS FELLMONGERY FIBERS FIBRO FIGURINE FILA FINNMARK FISHTAIL FLACKERY FLEWED FLOG FLOREATED FLOWERINESSES
    \\FOGEYISH FORWARDING FOULDERS FRETSAWS FRIGORIFICOS FRUICT FRUMP FUNGIBILITIES FURTIVENESS FUTCHEL GAITED GALLIASS GALLISISES GANGBUSTER
    \\GARBURES GATED GATH GAUCHERIES GENERALISE GIBBERELLIN GIVEABLE GLAMORED GLOBETROTS GLUMPY GONOPORE GOOFING GRACILIS GRASSED
    \\GRAVURES GRAY GRINNINGLY GRIZZLIES GUERNSEYS GURAMI GYPSIFEROUS HAGGLES HALAKISTS HALE HALOPHYTIC HAMPSTERS HANDBASKET HANDICAPPING
    \\HANDICAPS HANDSBREADTH HARBOURER HARDHANDEDNESS HARMONOGRAM HATCHELED HAYLAGE HAYSEED HAZIEST HEDONISTIC HELLERS HELLISHNESS HEMIPLEGICS HENNINS
    \\HEPTATHLON HERSTORY HIBERNICIZE HIEROGRAMS HOGNUTS HOOVER HORRIBLENESSES HORSEMEATS HUGGIER HUMANITARIANISM HUMBUGGED HYALIN HYGROCHASY HYPOBLASTS
    \\HYTHE ICEMAKERS IMMODERACY IMPERISHABLY IMPINGING IMPLODE INACTIVITY INCLUSIVITY INFIGHTINGS INJECTING INOSCULATES INSURANCE INTERISLAND INVISIBILITY
    \\IPOMOEA IRRADIANCE ISOLATIONISM JALOUSE JAZZES JELLY JOHNNY JOLLYBOAT JUMBUCK KIBLAS KITLING KITTULS KNOPPED KREESING
    \\KUGELS LALLED LANEWAY LANGUE LATHERY LATTICE LAWMEN LEAVIEST LEPIDOPTEROUS LEUCOMA LIBIDINAL LIGNOSES LINALOOLS LIONEL
    \\LIVABLENESSES LOIDS LOINCLOTH LYNXLIKE MACHINEMAN MACROCEPHALY MACROPHYLA MALADROITLY MALLOW MANA MANAIA MANDIRAS MANGINESSES MANGROVE
    \\MANHUNTER MANS MAQUILLAGE MARROWY MATACHINA MATROCLINOUS MATTIES MEFF MEGAPARSECS MEGRIMS MENE METEOROLITE MICROCOMPUTING MICRONUCLEUSES
    \\MILITANCIES MINATORIALLY MISADAPTS MISCORRECTIONS MISCREATION MISGOVERNMENTS MISINFORMATION MISLIKING MISSTYLE MISTRANSLATION MITIGATORS MIZZLIER MODERNLY MODEST
    \\MOIDERS MONGRELIZED MONOECIES MONOMANIACAL MONOSTICHIC MORONICALLY MUDIRIA MULTILOBE MUSLINED NEONATICIDES NEOPLASTY NERVURATIONS NETFULS NETTLES
    \\NIEFS NIGGERY NINETIES NOCTILUCENCES NONCARCINOGENS NONFULFILLMENTS NONGLOSSY NONINSECTS NONPREGNANT NONRECOGNITION NONRESIDENCES NOVELIZERS NULLIFICATION NUTTINGS
    \\OBLIVIONS OCCIDENTALISTS OFTEST OMNIPOTENTS OPSONISATIONS ORGANOGENESIS OSTRACIZERS OUTFLOW OUTLUSTRE OUTRANKED OUTREACH OUTSAILED OUTSWELLS OUTWARS
    \\OVERANALYSES OVERBEARINGLY OVEREYING OVERINFLATION OVERLANDERS OVERLAX OVERMULTITUDES OVERSOWS PALLETISATION PAMPHLET PAPERBACKER PAPISTRIES PARALLELLY PARAPHRAXES
    \\PARASITICALNESS PARQUETRY PATHOLOGISTS PATRIOT PEDESTRIANISE PEMOLINE PERIODICALISTS PERSICOTS PERSISTINGLY PERSPIRATED PETROGRAPHIES PHARAOH PHARMACOPEIAS PHOSPHORUS
    \\PHOSSY PHYLOGENESES PICKADILLS PIGBOAT PIGNERATING PIPI PISO PIXILATION PLANETOLOGIES PLATIEST PLEBEIANIZED PODGE POLARIZES POLYTHEISTIC
    \\PONTONS POONCING PORCUPINES POSTERIZATION POSTWOMAN POUNCET POURED POURS PRECEESE PRECIPITANTS PRECOGNOSCING PREDNISONES PREPUPA PRESELECT
    \\PRISSINESSES PRIVATENESSES PROBIT PROFILIST PROMINENTNESS PROPANEDIOIC PROTECTORIES PROTRUSILE PROVOCATIONS PSYCHONOMICS PSYOPS PUNKIN PYOGENESIS QUESTIONARY
    \\QUINTETTES QUIPPED RADIOACTIVE RADIOECOLOGY RATELS RATHOUSE RATOONERS REBS RECAPITALISES RECAPITULATORY RECOLLECTEDNESS RECTRICES RECULED REDSHIRT
    \\REENLARGING REFLATING REGREETED REHANDLES REIKI REJECTED RELIGIONIST REMOTES RENAGUE RESERVOIRS RESTABILISED RESUMPTION RETAINING RETEAMING
    \\RETIRALS RHEAS RHINESTONE RIGOLS RISE ROLLICKY RONE ROOTS ROSTRATE ROTIS ROUSE RUBBERED RUBIED RUNNIER
    \\RUSTICWORK SAGEBRUSH SAHIBAH SAINTHOOD SANTALINS SAPPY SARANS SATELESS SATES SAUCES SCALLOPS SCAMPI SCHMALTZY SCORSING
    \\SCREECHES SCREWIER SCRIMMAGER SCROGGIN SCRUBBIER SCUTCHED SCYPHATE SEAMANLIKE SEATBACKS SECURITAN SEDUCTOR SELFSAME SELSYN SENARIES
    \\SEPARATIVE SEVERITIES SEXTUPLETS SHAMOYED SHAMS SHANTYMAN SHATTERER SHELLFISH SHOCKABILITY SIBILATION SIDEPIECE SIREE SIRENISING SKETCHPAD
    \\SKIDLIDS SKINKED SKIPPIEST SKRIED SLOGGER SLOKEN SLUICED SMELTER SMOLT SNIFTERING SNOWBOARD SNOWSURFINGS SNUFFLY SODOMISTS
    \\SOLDERABILITY SONNES SOUNDBOARDS SPHERIC SPICEBUSH SPLURGES SPOILFIVE STEPFAMILIES STIPULES STOCKBROKING STRAFED STRETTAS STRIPT STROPHANTHUS
    \\STUCCO STUPING SUBLEVEL SUBTILIZING SUBVIRUS SUCCOURLESS SULFURETED SUMMARISERS SURFIER SURTOUT SWALLETS SYNTONIZED TABANIDS TARADIDDLE
    \\TARP TASLET TASTEMAKER TEACUPFULS TEENTSY TEGGS TENTLESS TEPHILLAH TESSERA TESTIFICATIONS TETRACTS THANKFULLEST THEEKING THESPIANS
    \\THICKENING THINKING THURIFEROUS TIGHT TILDES TIPPYTOE TIPSIFIED TIRRIVEE TITBITS TOBOGGANING TORMENTS TOROIDAL TOWNLING TRANSFUSABLE
    \\TRANSHUMING TRENAIL TRICHLORFON TRICHLORIDE TRIJUGOUS TRISTEZA TROUSERED TRUSTWORTHILY TUKTOO TUMULT TUTORIAL TWEAKIEST UMBRATED UNAMAZED
    \\UNDARING UNDEFIDE UNDEPRESSED UNDEREDUCATED UNDERSECRETARY UNGENEROSITIES UNINURED UNJADED UNLACES UNPINKED UNPROFESSED UNPRONOUNCEABLE UNREDY UNRELUCTANT
    \\UNSANDALLED UNVIRTUE UPGAZES UPGAZING UPMARKET UPPER URACHUSES URBANISTS VAPORABILITY VAROOMING VAULTLIKE VELVETED VERNAL VICARAGE
    \\VIHARA VINTRIES VITALLY WADDING WAINSCOTTING WANGUN WATCHFUL WATERPROOFS WEATHERISING WEATHEROMETERS WEEPIE WHAISLING WHIDS WHITISHNESS
    \\WHOOPERS WINTERLESS WIVES WIZZENS WOMANNESSES WOODPIGEONS WORMINESS WORTLES WRATHED WRITERSHIP XEROXED YAPPED YICKERING YODELLERS
    \\ZAFFARS ZOCCOLO
;

const Entry = struct {
    key: [8]u16,
    score: i32,

    fn lessThan(_: void, a: Entry, b: Entry) bool {
        return switch (std.mem.order(u16, &a.key, &b.key)) {
            .lt => true,
            .gt => false,
            .eq => a.score < b.score,
        };
    }
};

/// Sorted (placement, score) pairs with repeats removed.
fn entries(gpa: std.mem.Allocator, moves: []const Move) ![]Entry {
    var list: std.ArrayList(Entry) = .empty;
    errdefer list.deinit(gpa);
    for (moves) |*m| try list.append(gpa, .{ .key = m.key(), .score = m.score });
    std.sort.pdq(Entry, list.items, {}, Entry.lessThan);
    var n: usize = 0;
    for (list.items) |e| {
        if (n > 0 and std.mem.eql(u16, &list.items[n - 1].key, &e.key)) {
            try std.testing.expectEqual(list.items[n - 1].score, e.score);
            continue;
        }
        list.items[n] = e;
        n += 1;
    }
    list.shrinkRetainingCapacity(n);
    return list.toOwnedSlice(gpa);
}

/// True if each blank in `m` stands for a letter the natural tiles in
/// `supply` cannot cover, and no more blanks are used than that.
fn blanksNeeded(m: *const Move, supply: *const [27]u8) bool {
    var uses: [26]u8 = @splat(0);
    var blanks: [26]u8 = @splat(0);
    var it = m.new_mask;
    while (it != 0) : (it &= it - 1) {
        const c = m.tiles[@ctz(it)];
        uses[rules.letterOf(c)] += 1;
        if (rules.isBlank(c)) blanks[rules.letterOf(c)] += 1;
    }
    for (0..26) |l| {
        if (blanks[l] != uses[l] -| supply[l]) return false;
    }
    return true;
}

fn playRandomGames(r: rules.Rules, games: usize, seed: u64) !void {
    const gpa = std.testing.allocator;
    var lex = try lexicon.build(gpa, test_words);
    defer lex.deinit(gpa);
    var words = try csp.WordSets.init(gpa, test_words);
    defer words.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    var gad = movegen.Collector{ .gpa = gpa };
    defer gad.deinit();
    var ref = movegen.Collector{ .gpa = gpa };
    defer ref.deinit();
    var needed = movegen.Collector{ .gpa = gpa };
    defer needed.deinit();
    var turns: std.ArrayList(game.Turn) = .empty;
    defer turns.deinit(gpa);
    const realized = try gpa.create(game.Realized);
    defer gpa.destroy(realized);

    var positions: usize = 0;
    for (0..games) |_| {
        var g = game.Game.init(r);
        turns.clearRetainingCapacity();
        while (!g.over()) {
            const sup = g.supply();
            gad.moves.clearRetainingCapacity();
            ref.moves.clearRetainingCapacity();
            needed.moves.clearRetainingCapacity();
            movegen.generate(movegen.Collector, &gad, &g.pos, &lex, r, sup);
            const halo = csp.computeHalo(&words, &g.pos);
            try csp.generate(movegen.Collector, &ref, gpa, &words, &g.pos, &halo, r, sup);
            try std.testing.expect(!gad.failed and !ref.failed);

            const a = try entries(gpa, gad.moves.items);
            defer gpa.free(a);
            const b = try entries(gpa, ref.moves.items);
            defer gpa.free(b);
            // The GADDAG generator never repeats a play.
            try std.testing.expectEqual(gad.moves.items.len, a.len);
            try std.testing.expectEqual(b.len, a.len);
            for (a, b) |x, y| {
                try std.testing.expectEqualSlices(u16, &y.key, &x.key);
                try std.testing.expectEqual(y.score, x.score);
            }
            for (gad.moves.items) |*m| {
                try std.testing.expectEqual(g.pos.scoreMove(m, r), m.score);
                try std.testing.expect(m.placed <= sup.max_tiles);
            }
            // The as-needed blank mode yields exactly the plays that use
            // blanks only where they must.
            var as_needed = sup;
            as_needed.blanks = .as_needed;
            movegen.generate(movegen.Collector, &needed, &g.pos, &lex, r, as_needed);
            var expected: usize = 0;
            for (gad.moves.items) |*m| expected += @intFromBool(blanksNeeded(m, &sup.counts));
            try std.testing.expectEqual(expected, needed.moves.items.len);
            for (needed.moves.items) |*m| try std.testing.expect(blanksNeeded(m, &sup.counts));
            positions += 1;

            if (gad.moves.items.len == 0) {
                g.pass();
                try turns.append(gpa, .pass);
                continue;
            }
            const pick = gad.moves.items[rng.uintLessThan(usize, gad.moves.items.len)];
            // Plays round-trip through the text format.
            var buf: [64]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            try g.pos.writePlay(&pick, &w);
            const text = w.buffered();
            var tok = std.mem.tokenizeScalar(u8, text, ' ');
            const row = try std.fmt.parseInt(usize, tok.next().?, 10);
            const col = try std.fmt.parseInt(usize, tok.next().?, 10);
            const d: u1 = if (std.mem.eql(u8, tok.next().?, "across")) 0 else 1;
            const parsed = try g.pos.parsePlay(&lex, r, row, col, d, tok.next().?);
            try std.testing.expectEqualSlices(u16, &pick.key(), &parsed.key());
            try std.testing.expectEqual(pick.score, parsed.score);

            g.play(&lex, &pick);
            try turns.append(gpa, .{ .play = pick });
            var fresh = g.pos;
            fresh.recomputeCrossChecks(&lex);
            for (0..2) |dd| for (0..rules.NN) |i| {
                if (g.pos.cells[dd][i] != 0) continue;
                try std.testing.expectEqual(fresh.xmask[dd][i], g.pos.xmask[dd][i]);
                try std.testing.expectEqual(fresh.xsum[dd][i], g.pos.xsum[dd][i]);
            };
        }
        // The game replays with real racks and draws, to the same scores.
        const again = try game.realize(&lex, r, turns.items, realized);
        try std.testing.expectEqual(g.score, again.score);
    }
    try std.testing.expect(positions > games * 5);
}

test "GADDAG matches the CSP port: solitaire, no blanks (solver.py's rules)" {
    try playRandomGames(.{ .players = 1, .blanks = 0 }, 4, 1);
}

test "GADDAG matches the CSP port: two players, two blanks" {
    try playRandomGames(.{}, 2, 2);
}

test "GADDAG matches the CSP port: solitaire, two blanks, NASPA end" {
    try playRandomGames(.{ .players = 1, .end_rule = .naspa }, 2, 3);
}

test "every dictionary word is accepted and its GADDAG paths exist" {
    const gpa = std.testing.allocator;
    var lex = try lexicon.build(gpa, test_words);
    defer lex.deinit(gpa);
    var tok = std.mem.tokenizeAny(u8, test_words, " \n");
    var n: usize = 0;
    var tmp: [15]u5 = undefined;
    while (tok.next()) |w| {
        const word = lexicon.wordToLetters(w, &tmp);
        try std.testing.expect(lex.contains(word));
        // rev(prefix) SEP suffix for every split.
        for (1..word.len) |i| {
            var node = lex.root();
            var k = i;
            while (k > 0) {
                k -= 1;
                node = lex.child(node, word[k]);
            }
            node = lex.child(node, lexicon.SEP);
            for (word[i..]) |l| node = lex.child(node, l);
            try std.testing.expect(lexicon.accepts(node));
        }
        n += 1;
    }
    try std.testing.expectEqual(@as(u32, @intCast(n)), lex.word_count);
    // Words with one letter changed are mostly not words; check against the list.
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var set: std.StringHashMapUnmanaged(void) = .empty;
    defer set.deinit(gpa);
    tok = std.mem.tokenizeAny(u8, test_words, " \n");
    while (tok.next()) |w| try set.put(gpa, w, {});
    tok = std.mem.tokenizeAny(u8, test_words, " \n");
    while (tok.next()) |w| {
        var buf: [15]u8 = undefined;
        @memcpy(buf[0..w.len], w);
        buf[rng.uintLessThan(usize, w.len)] = 'A' + rng.uintLessThan(u8, 26);
        const expect = set.contains(buf[0..w.len]);
        try std.testing.expectEqual(expect, lex.contains(lexicon.wordToLetters(buf[0..w.len], &tmp)));
    }
}

test "first plays: count, centre coverage and a hand-scored example" {
    const gpa = std.testing.allocator;
    var lex = try lexicon.build(gpa, "JOKER\nJO\nOK\nKO\nOX\n");
    defer lex.deinit(gpa);
    const r = rules.Rules{ .blanks = 0 };
    const pos = Position.init(r);
    var col = movegen.Collector{ .gpa = gpa };
    defer col.deinit();
    movegen.generate(movegen.Collector, &col, &pos, &lex, r, .{ .counts = pos.pool, .max_tiles = 7 });
    // Each word, in each direction, at each offset that covers the centre.
    try std.testing.expectEqual(@as(usize, 2 * (5 + 2 + 2 + 2 + 2)), col.moves.items.len);
    // JOKER across from (7,3): J on the double letter at (7,3), the centre
    // doubles the word: (8*2 + 1 + 5 + 1 + 1) * 2 = 48.
    const m = try pos.parsePlay(&lex, r, 7, 3, 0, "JOKER");
    try std.testing.expectEqual(@as(i32, 48), m.score);
    var found = false;
    for (col.moves.items) |*x| {
        if (std.mem.eql(u16, &x.key(), &m.key())) {
            try std.testing.expectEqual(m.score, x.score);
            found = true;
        }
    }
    try std.testing.expect(found);
    try std.testing.expectError(error.NotConnected, pos.parsePlay(&lex, r, 0, 0, 0, "JOKER"));
}
