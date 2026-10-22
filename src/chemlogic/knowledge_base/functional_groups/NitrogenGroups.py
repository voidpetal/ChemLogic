from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class NitrogenGroups(KnowledgeBase):
    required_keys = ["layer_name", "param_size", "carbon", "oxygen", "nitrogen"]

    def create_template(self):
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_n_carbonyl")(V.N)
                <= R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R, V.N)
            ]
        )
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_n_quaternary")(V.N)
                <= (
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R3),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R4),
                    R.special.alldiff(V.N, V.R1, V.R2, V.R3, V.R4),
                )
            ]
        )

        # Defining amine group (R1-C-N(-R2)-R3)
        # TODO: this won't work for datasets with implicit hydrogens (primary, secondary amines)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_amine")(V.N)
                <= (
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.C, V.R1, V.R2),
                    ~R.hidden.get(f"{self.layer_name}_n_carbonyl")(V.N),
                    ~R.hidden.get(f"{self.layer_name}_n_quaternary")(V.N),
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3)
                <= (
                    R.get(self.carbon)(V.R1),
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R2, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R3, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.N, V.R1, V.R2, V.R3),
                )
            ]
        )

        # Defining quaternary ammonium ion (R1-N(-R2)(-R3)-R4)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_quaternary_ammonium")(V.N)
                <= (
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.C, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.N, V.R1, V.R2, V.R3, V.C),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_quat_ammonion")(V.N)
                <= R.get(f"{self.layer_name}_quaternary_ammonium")(V.N)
            ]
        )

        # Defining amide group (R-C(=O)-N(-R1)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_amide")(V.N)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R, V.N),
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.C, V.R1, V.R2),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C, V.O, V.N),
                )
            ]
        )

        # Defining imine group (R1-C(=N-R)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imine")(V.N)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N, V.B),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R2, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C, V.N),
                )
            ]
        )

        # Defining imide group (R1-C(=O)-N(-R)-C(=O)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imide")(V.N)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.C2),
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C1, V.O1, V.R1, V.N),
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C2, V.O2, V.R2, V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C1, V.C2, V.N, V.O1, V.O2),
                )
            ]
        )

        # Defining a nitrile (R-C≡N); the carrier is the nitrile carbon C.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrile")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_triple_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.special.alldiff(V.C, V.N, V.R),
                )
            ]
        )

        # Defining an isocyanide (R-N≡C); the carrier is the attached nitrogen N.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_isocyanide")(V.N)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_triple_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.special.alldiff(V.C, V.N, V.R),
                )
            ]
        )

        # Defining a hydrazine (N-N); each nitrogen atom is a carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydrazine")(V.N1)
                <= (
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N1, V.N2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.N1, V.N2),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydrazine")(V.N2)
                <= (
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N1, V.N2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.N1, V.N2),
                )
            ]
        )

        # Defining a hydrazone (C=N-N); the imine nitrogen is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydrazone")(V.N1)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_hydrazine")(V.N1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.special.alldiff(V.C, V.N1),
                )
            ]
        )

        # Defining an oxime (C=N-O); the imine nitrogen is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxime")(V.N)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.O, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.special.alldiff(V.C, V.N, V.O),
                )
            ]
        )

        # Shared amidine core (C(=N)-N) for amidine, guanidine, and urea motifs.
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_amidine_core")(V.C, V.N1, V.N2)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.N2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.special.alldiff(V.C, V.N1, V.N2),
                )
            ]
        )
        # The imino nitrogen is the amidine carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_amidine")(V.N1)
                <= R.hidden.get(f"{self.layer_name}_amidine_core")(V.C, V.N1, V.N2)
            ]
        )

        # Defining a guanidine (C(=N)(-N)-N); the imino nitrogen is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_guanidine")(V.N1)
                <= (
                    R.get(self.nitrogen)(V.N3),
                    R.hidden.get(f"{self.layer_name}_amidine_core")(V.C, V.N1, V.N2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.N3, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.C, V.N3, V.N1, V.N2),
                )
            ]
        )

        # Defining a urea (N-C(=O)-N); the carbonyl carbon is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_urea")(V.C)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.N1, V.N2),
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_urea")(V.C)
                <= (
                    R.hidden.get(f"{self.layer_name}_amidine_core")(V.C, V.N1, V.N2),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.special.alldiff(V.C, V.O, V.N1, V.N2),
                )
            ]
        )

        # Defining a nitroso group (R-N=O); the nitroso nitrogen is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitroso")(V.N)
                <= (
                    R.get(self.carbon)(V.R),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N, V.O, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    ~R.get(f"{self.layer_name}_nitro_group")(V.N),
                    R.special.alldiff(V.R, V.N, V.O),
                )
            ]
        )

        # Defining azide group (R-N=N=N)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_azide")(V.N1)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.get(self.nitrogen)(V.N3),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.N1, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N1, V.N2, V.B2),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N2, V.N3, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.C, V.N1, V.N2, V.N3),
                )
            ]
        )

        # Defining azo group (R1-N=N-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_azo")(V.N1, V.N2)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.get(self.carbon)(V.C2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C1, V.N1, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N1, V.N2, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N2, V.C2, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.C1, V.C2, V.N1, V.N2),
                )
            ]
        )

        # Defining cyanate group (R-O-C≡N)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_cyanate")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_triple_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.C, V.N, V.O, V.R),
                )
            ]
        )

        # Defining isocyanate group (R-N=C=O)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_isocyanate")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.O, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.R, V.C, V.O, V.N),
                )
            ]
        )

        # Defining nitro group (R-N(=O)-O)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitro_group")(V.N)
                <= (
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N, V.O1, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.O2, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.R, V.N, V.O1, V.O2),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitro")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_nitro_group")(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.N, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                )
            ]
        )

        # Defining nitrate group (R-O-N(=O)-O)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrate")(V.N)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N, V.O1, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.O2, V.B3),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.C, V.O, V.N, V.O1, V.O2),
                )
            ]
        )

        # Defining carbamate group (R-O-C(=O)-N(-R1)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbamate")(V.N)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O1, V.O2, V.N),
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.C, V.R1, V.R2),
                    R.get(self.oxygen)(V.O2),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O2, V.R, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C, V.O1, V.O2, V.N),
                )
            ]
        )

        # Defining an aziridine ring (three-membered C-C-N ring)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aziridine")(V.N)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.carbon)(V.C2),
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C1, V.C2, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.C1, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.C2, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B3),
                    R.special.alldiff(V.C1, V.C2, V.N),
                )
            ]
        )

        # Aggregating the nitrogen groups
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_amine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_quaternary_ammonium")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_amide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_nitrile")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_isocyanide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_hydrazine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_hydrazone")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_oxime")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_amidine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_guanidine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_urea")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_nitroso")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_imine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_azide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_imide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.N1)
                <= R.get(f"{self.layer_name}_azo")(V.N1, V.N2)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.N2)
                <= R.get(f"{self.layer_name}_azo")(V.N1, V.N2)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_isocyanate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_cyanate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_nitro")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_nitrate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_carbamate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_aziridine")(V.X)[self.param_size]
            ]
        )
